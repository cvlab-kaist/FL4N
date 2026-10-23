"""FL4N runner: protocol loaders -> pose-free (or posed) conditioning -> render -> protocol scorer."""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

BENCH = Path(__file__).resolve().parent.parent
ROOT = BENCH.parent
sys.path.insert(0, str(ROOT))
for _d in ("common", "datasets", "runners"):
    sys.path.insert(0, str(BENCH / _d))

from models.checkpoint import FL4N_CKPT, load_checkpoint  # noqa: E402
from data import normalization  # noqa: E402
from protocol_score import Scorer  # noqa: E402
import bench_proto_common as C  # noqa: E402
from bench_paths import OUT as OUT_ROOT, dataset  # noqa: E402

NV = dataset("NVIDIA")
VGGT_POSE_DIR = os.environ.get("FL4N_VGGT_POSE_DIR", "")
POSED = False


def _proto_loader(mod_name, load_attr):
    def f(scene):
        d = getattr(__import__(mod_name), load_attr)(scene)
        return d["cond"], C.flatten_items(d["targets"])
    return f


def _kubric():
    import bench_ds_kubric4d as m
    return dict(scenes=list(m.KUBRIC4D_SCENES), load=_proto_loader("bench_ds_kubric4d", "load_kubric4d_scene"), render_hw=(336, 504))


def _camxtime():
    import bench_ds_camxtime as m
    return dict(scenes=list(m.list_test_scenes()), load=_proto_loader("bench_ds_camxtime", "load_camxtime_scene"), render_hw=(504, 504))


def _charge():
    import bench_ds_charge as m
    return dict(scenes=list(m.SCENES), load=_proto_loader("bench_ds_charge", "load_charge_scene"), render_hw=(448, 1064))


def _n3dv():
    import bench_ds_n3dv as m
    return dict(scenes=list(m.N3DV_SCENES), load=_proto_loader("bench_ds_n3dv", "load_n3dv_scene"), render_hw=(504, 672))


def _read_calib_nvidia(scene_root, cam):
    K = np.loadtxt(scene_root / "calibration" / f"cam{cam + 1:02d}" / "intrinsic.txt", dtype=np.float32)
    ext = np.loadtxt(scene_root / "calibration" / f"cam{cam + 1:02d}" / "extrinsic.txt", dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = ext[1:].T
    c2w[:3, 3] = ext[0]
    return K, c2w


def _load_nvidia_ctvs(scene):
    """CTVS on NVIDIA: 12 monocular inputs on the camera x time diagonal, all off-diagonal cells as targets."""
    sr = NV / scene
    gt_root = sr / "multiview_GT"
    if not (gt_root / "00000001").is_dir():
        gt_root = gt_root / "multiview_GT"
    calib = [_read_calib_nvidia(sr, c) for c in range(12)]
    cond = []
    for c in range(12):
        img = cv2.cvtColor(cv2.imread(str(sr / "input_images" / f"cam{c + 1:02d}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        cond.append({"image": img, "K": calib[c][0], "c2w": calib[c][1], "frame": c})
    items = []
    for cam in range(12):
        for t in range(12):
            if cam == t:
                continue
            gt = np.asarray(Image.open(gt_root / f"{t + 1:08d}" / f"cam{cam + 1:02d}.jpg").convert("RGB"), np.float32) / 255.0
            items.append(("all", {"c2w": calib[cam][1], "K": calib[cam][0], "gt": gt, "dyn_mask": None, "frame": t, "view": cam}))
    return cond, items


def _nvidia():
    import bench_ds_nvidia as m
    if os.environ.get("BENCH_TABLE2"):
        return dict(scenes=list(m.NVIDIA_SCENES), load=_proto_loader("bench_ds_nvidia", "load_nvidia_nvs"), render_hw=(504, 896))
    return dict(scenes=list(m.NVIDIA_SCENES), load=_load_nvidia_ctvs, render_hw=(504, 896))


def _adt():
    import bench_ds_adt as m
    return dict(scenes=list(m.ADT_SCENES), load=_proto_loader("bench_ds_adt", "load_adt_scene"), render_hw=(512, 512))


def _dycheck():
    import bench_ds_dycheck as m
    return dict(scenes=list(m.DYCHECK_SCENES), load=_proto_loader("bench_ds_dycheck", "load_dycheck_scene"), render_hw=(672, 504))


def _tum():
    import bench_ds_tum as m
    return dict(scenes=list(m.TUM_SCENES), load=_proto_loader("bench_ds_tum", "load_tum_scene"), render_hw=(336, 448))


DATASETS = {"kubric4d": _kubric, "camxtime": _camxtime, "charge": _charge, "n3dv": _n3dv, "nvidia": _nvidia,
            "adt": _adt, "dycheck": _dycheck, "tum": _tum}


def build_inputs(cond, items, render_hw):
    """(images, rays, cam_token, tau, K) over [cond | targets] at render res; target image slots are zeros."""
    RH, RW = render_hw
    imgs, c2ws, intr, frames = [], [], [], []
    K = len(cond)
    for c in cond:
        Ko = C.scale_K(c["K"], c["image"].shape[0], c["image"].shape[1], RH, RW)
        img = cv2.resize(np.clip(c["image"], 0, 1).astype(np.float32), (RW, RH), interpolation=cv2.INTER_AREA)
        imgs.append(np.transpose(img, (2, 0, 1)))
        c2ws.append(np.asarray(c["c2w"], np.float32))
        frames.append(int(c["frame"]))
        intr.append([Ko[0, 0], Ko[1, 1], Ko[0, 2], Ko[1, 2]])
    for _, it in items:
        Ko = C.scale_K(it["K"], it["gt"].shape[0], it["gt"].shape[1], RH, RW)
        imgs.append(np.zeros((3, RH, RW), np.float32))
        c2ws.append(np.asarray(it["c2w"], np.float32))
        frames.append(int(it["frame"]))
        intr.append([Ko[0, 0], Ko[1, 1], Ko[0, 2], Ko[1, 2]])
    images = torch.from_numpy(np.stack(imgs)).float()
    c2w_norm, cam_scale = normalization.normalize_extrinsics(torch.from_numpy(np.stack(c2ws)).float(), num_cond_views=K)
    intr_t = torch.tensor(intr, dtype=torch.float32)
    cam_enc, cam_token = normalization.build_cam_cond(c2w_norm, intr_t, num_cond_views=K, tgt_hw=(RH, RW),
                                                      camera_scale=cam_scale, zero_cond_pose=not POSED)
    fmin = min(frames[:K])
    tau = torch.tensor([f - fmin for f in frames], dtype=torch.float32)
    return images, cam_enc, cam_token, tau, K


def render(model, images, rays, camtok, tau, K, chunk, dev):
    """Render target views (indices >= K) in chunks; returns [T,3,H,W] cpu float in [0,1]."""
    out = images[0].clone()
    n = out.shape[0] - K
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, n, chunk):
            ci = torch.arange(s, min(s + chunk, n), device=dev)
            sel = torch.cat([torch.arange(K, device=dev), K + ci])
            out[K + ci] = model(images[:, sel], rays[:, sel], camtok[:, sel], num_cond_views=K, tau=tau[:, sel])[0, K:].float()
    return out[K:].cpu()


def apply_vggt_poses(ds, scene, cond, items):
    """Replace cond+target c2w (and K with FL4N_VGGT_INTR=1) by VGGT estimates, computed once and cached."""
    f = os.path.join(VGGT_POSE_DIR, f"{ds}__{scene}.npz")
    if not os.path.exists(f):
        import vggt_pose
        cond_images = None
        if ds == "nvidia" and not os.environ.get("BENCH_TABLE2"):
            import bench_ds_nvidia
            cond_images = [c["image"] for c in bench_ds_nvidia.load_nvidia_ctvs(scene)["cond"]]
        c2w, K = vggt_pose.scene_cameras(cond, items, cond_images)
        os.makedirs(VGGT_POSE_DIR, exist_ok=True)
        np.savez_compressed(f, c2w=c2w, K=K, n_cond=len(cond), native_hw=np.array(cond[0]["image"].shape[:2]))
    z = np.load(f)
    PS, K = z["c2w"], int(z["n_cond"])
    if len(cond) != K or len(items) != PS.shape[0] - K:
        raise RuntimeError(f"vggt pose cache size mismatch for {ds}/{scene}")
    VK = z["K"] if ("K" in z.files and os.environ.get("FL4N_VGGT_INTR") == "1") else None
    cond = [{**c, "c2w": PS[i], **({"K": VK[i]} if VK is not None else {})} for i, c in enumerate(cond)]
    items = [(g, {**d, "c2w": PS[K + i], **({"K": VK[K + i]} if VK is not None else {})}) for i, (g, d) in enumerate(items)]
    return cond, items


def run_scene(model, scorer, cfg, ds, scene, out_dir, chunk, dev):
    cond, items = cfg["load"](scene)
    if os.environ.get("FL4N_VGGT_POSE") == "1":
        cond, items = apply_vggt_poses(ds, scene, cond, items)
    render_hw = C.render_hw_override(items[0][1]["gt"].shape[:2], cfg["render_hw"], 8)
    images, cam_enc, cam_token, tau, K = build_inputs(cond, items, render_hw)
    images, cam_enc, cam_token, tau = (x[None].to(dev) for x in (images, cam_enc, cam_token, tau))
    preds = [np.transpose(p.numpy(), (1, 2, 0)) for p in render(model, images, cam_enc, cam_token, tau, K, chunk, dev)]
    return C.score_and_save(items, preds, scorer, out_dir / scene, scene, render_hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--ckpt", default=FL4N_CKPT)
    ap.add_argument("--posed", action="store_true", help="feed the conditioning camera poses (default: pose-free)")
    ap.add_argument("--out-name", default=None, help="output folder name (default: fl4n or fl4n_posed)")
    ap.add_argument("--chunk", type=int, default=4)
    args = ap.parse_args()
    global POSED
    POSED = args.posed
    if not (args.dataset or args.all):
        ap.error("pass --dataset <name> or --all")
    dev = "cuda"
    model, step = load_checkpoint(args.ckpt, dev)
    print(f"[fl4n] loaded {Path(args.ckpt).name} | step={step} | posed={POSED}", flush=True)
    scorer = Scorer(device=dev)
    for ds in (list(DATASETS) if args.all else [args.dataset]):
        cfg = DATASETS[ds]()
        out_dir = OUT_ROOT / ds / (args.out_name or ("fl4n_posed" if POSED else "fl4n"))
        out_dir.mkdir(parents=True, exist_ok=True)
        per_scene = {}
        for sc in (args.scenes or cfg["scenes"]):
            try:
                per_scene[sc] = run_scene(model, scorer, cfg, ds, sc, out_dir, args.chunk, dev)
                print(f"[{ds}] {sc}: PSNR {per_scene[sc]['all']['psnr']:.3f}", flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[{ds}] {sc}: FAILED {e}", flush=True)
            torch.cuda.empty_cache()
        if per_scene:
            framing = {"render_hw": list(cfg["render_hw"]), "input": "full native-aspect resize (no crop)",
                       "score_res": "native GT, pred bilinear-upsampled", "lpips": "vgg",
                       "posed_cond": POSED, "vggt_pose": os.environ.get("FL4N_VGGT_POSE") == "1",
                       "pose_norm": "recenter cam0 + camera-baseline scale", "time_token": "frame offset from the earliest cond frame",
                       "ckpt": args.ckpt, "global_step": step}
            summary = C.write_summary(out_dir, "FL4N", ds, per_scene, framing, ("all",))
            print(f"[{ds}] OVERALL", {k: round(summary["overall"][k], 4) for k in C.METRIC_KEYS}, flush=True)


if __name__ == "__main__":
    main()
