"""Flow-based dynamic masks for training (CamXTime, Kubric4D train), from WAFT optical flow.

Frame t is marked dynamic where |flow(I_t -> I_{t+gap})| > threshold (I_{t-gap} for the last `gap` frames).

Output (read by data/sources/{camxtime,kubric4d}.py), bit-packed along width, bitorder="little":
  CamXTime  <out>/<Scene>/camera-trajectory-XX.npz   members mask_camNNN [T, 336, 64]  (rows 88:424)
  Kubric4D  <out>/<scene>.npz                        members mask_vN     [T, H, W/8]

Usage:
  python tools/make_masks.py --dataset camxtime --threshold 0.4 --gap 16 --out $FL4N_DATA/masks/camxtime
  python tools/make_masks.py --dataset kubric4d --threshold 5.0 --gap 4  --out $FL4N_DATA/masks/kubric4d_gap4
  python tools/make_masks.py --dataset kubric4d --threshold 8.0 --gap 8  --out $FL4N_DATA/masks/kubric4d_gap8
Add --shard i --num-shards N to split scenes over GPUs; finished archives are skipped, so rerunning resumes.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

DATA = Path(os.environ.get("FL4N_DATA", "data"))
CXT_CROP = slice(88, 424)


def to56(x):
    return max(56, int(round(x / 56.0)) * 56)


def build_waft(waft_root, cfg, ckpt):
    sys.path.insert(0, str(waft_root))
    from config.parser import json_to_args
    from inference_tools import InferenceWrapper
    from model import fetch_model

    cfg, ckpt, cwd = os.path.abspath(cfg), os.path.abspath(ckpt), os.getcwd()
    args = json_to_args(cfg)
    os.chdir(waft_root)  # WAFT loads depth-anything-ckpts/ relative to its repo root
    try:
        model = fetch_model(args)
    finally:
        os.chdir(cwd)
    miss, unexp = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    if miss or unexp:
        raise SystemExit(f"checkpoint/config mismatch: missing={len(miss)} unexpected={len(unexp)}")
    return InferenceWrapper(model.cuda().eval(), scale=0.0, train_size=args.image_size,
                            pad_to_train_size=False, tiling=False)


@torch.no_grad()
def dyn_masks(waft, frames, gap, thr, batch, amp):
    """frames: list of [H, W, 3] float RGB in [0, 255] -> bool [T, H, W]."""
    H, W = frames[0].shape[:2]
    Ht, Wt = to56(H), to56(W)
    n = len(frames)
    ten = torch.from_numpy(np.stack([cv2.resize(f, (Wt, Ht)) for f in frames])).float().permute(0, 3, 1, 2).cuda()
    pairs = [(t, t + gap) if t + gap < n else (t, t - gap) for t in range(n)]
    masks = np.zeros((n, H, W), bool)
    for s in range(0, n, batch):
        chunk = pairs[s: s + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            f = waft.calc_flow(ten[[a for a, _ in chunk]], ten[[b for _, b in chunk]])["flow"][-1]
        f = f.float()
        if (Ht, Wt) != (H, W):
            f = torch.nn.functional.interpolate(f, size=(H, W), mode="bilinear", align_corners=True)
            f[:, 0] *= W / Wt
            f[:, 1] *= H / Ht
        masks[s: s + len(chunk)] = (f.norm(dim=1) > thr).cpu().numpy()
    return masks


def pack(m):
    return np.packbits(m, axis=-1, bitorder="little")


def kubric_frames(view_dir):
    """Load a Kubric view's RGB frames, replacing unreadable PNGs with the previous readable frame."""
    out = []
    for p in sorted(view_dir.glob("rgba_*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        out.append(None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32))
    good = [f for f in out if f is not None]
    if not good:
        return []
    prev = good[0]
    for i, f in enumerate(out):
        out[i] = prev = (f if f is not None else prev.copy())
    return out


def video_frames(path):
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32)[CXT_CROP])
    cap.release()
    return out


def run_kubric(waft, root, out, scenes, a):
    for i, sc in enumerate(scenes):
        dst = out / f"{sc}.npz"
        if dst.exists():
            continue
        t0, arrs = time.time(), {}
        for vd in sorted((root / sc).glob("frames_p0_v*"), key=lambda p: int(p.name.split("_v")[1])):
            fr = kubric_frames(vd)
            if fr:
                m = dyn_masks(waft, fr, a.gap, a.threshold, a.batch, a.amp)
                arrs[f"mask_v{vd.name.split('_v')[1]}"] = pack(m)
        if not arrs:
            continue
        views = sorted(int(k.split("_v")[1]) for k in arrs)
        np.savez_compressed(dst, scene=sc, threshold_px=a.threshold, gap=a.gap, views=np.array(views),
                            mask_shape_thw=np.array(m.shape), **arrs)
        print(f"[{i + 1}/{len(scenes)}] {sc} views={len(arrs)} {time.time() - t0:.0f}s", flush=True)


def run_camxtime(waft, root, out, scenes, a):
    for i, sc in enumerate(scenes):
        for tdir in sorted((root / sc).glob("camera-trajectory-*")):
            dst = out / sc / f"{tdir.name}.npz"
            if dst.exists():
                continue
            t0, arrs, cams = time.time(), {}, []
            for vid in sorted(tdir.glob("cam*_full_motion.mp4")):
                fr = video_frames(vid)
                if fr:
                    cam = int(vid.name[3:6])
                    m = dyn_masks(waft, fr, a.gap, a.threshold, a.batch, a.amp)
                    arrs[f"mask_cam{cam:03d}"] = pack(m)
                    cams.append(cam)
            if not arrs:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, scene=sc, trajectory=int(tdir.name[-2:]), threshold_px=a.threshold, gap=a.gap,
                                camera_numbers=np.array(cams), mask_shape_thw=np.array(m.shape), **arrs)
            print(f"[{i + 1}/{len(scenes)}] {sc}/{tdir.name} cams={len(arrs)} {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["camxtime", "kubric4d"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, required=True, help="flow magnitude in pixels")
    ap.add_argument("--gap", type=int, required=True, help="frame gap of the flow pair")
    ap.add_argument("--waft-root", default="third_party/WAFT", help="clone of github.com/princeton-vl/WAFT")
    ap.add_argument("--cfg", default=None, help="default: <waft-root>/config/a1/tar-c-t.json")
    ap.add_argument("--ckpt", default="checkpoints/waft_a1_adaptation.pth")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="disable bf16 autocast")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()

    root = DATA / ("CamxTime" if a.dataset == "camxtime" else "Kubric4D/train")
    scenes = sorted(p.name for p in root.iterdir() if p.is_dir())[a.shard:: a.num_shards]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.json").write_text(json.dumps(
        {"dataset": a.dataset, "threshold_px": a.threshold, "gap": a.gap, "amp_bf16": a.amp}, indent=2))

    waft_root = Path(a.waft_root).resolve()
    waft = build_waft(waft_root, a.cfg or str(waft_root / "config/a1/tar-c-t.json"), a.ckpt)
    print(f"{a.dataset} shard {a.shard}/{a.num_shards}: {len(scenes)} scenes, thr={a.threshold}px gap={a.gap}", flush=True)
    (run_camxtime if a.dataset == "camxtime" else run_kubric)(waft, root, out, scenes, a)


if __name__ == "__main__":
    main()
