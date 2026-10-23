#!/usr/bin/env python3
"""MoVieS runner: protocol loaders -> MoVieS reconstruct/render -> protocol scorer."""
import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

_B = Path(__file__).resolve().parent.parent       # benchmark/
for _d in ("common", "datasets", "runners"):
    sys.path.insert(0, str(_B / _d))
from bench_paths import BASELINES, OUT, weights  # noqa: E402

REPO = BASELINES / "MoVieS"
sys.path.insert(3, str(REPO))
os.chdir(REPO)
if "Wand" not in sys.modules:
    _wand = types.ModuleType("Wand")
    _wand.Image = _wand.Video = object
    sys.modules["Wand"] = _wand
from src.models import SplatRecon  # noqa: E402
from src.options import opt_dict  # noqa: E402
from src.utils import unproject_depth  # noqa: E402
from protocol_score import Scorer  # noqa: E402
import bench_proto_common as C  # noqa: E402

CKPT = weights("movies", "movies_ckpt.safetensors")
OUT_ROOT = OUT

_LEGACY_FIX = os.environ.get("MOVIES_FIX", "") not in ("", "0", "false", "False", "no")
POSE_SCALE = os.environ.get("MOVIES_POSE_SCALE", "pc" if _LEGACY_FIX else "baseline").lower()
TIME_SPAN = os.environ.get("MOVIES_TIME", "union").lower()


def _pc_scale(model, images, cond_c2ws, canon, input_intr, input_times, device, rig_scale):
    """Rescale poses so unprojected input points have unit mean distance from cam0."""
    try:
        cond_canon = C.apply_norm_movies(cond_c2ws, canon, 1.0)  # (N, 4, 4)
        ot = input_times[:, :1]  # static backbone is time-independent; 1 dummy out-time
        scale = 1.0
        for _ in range(2):  # initial estimate + one refinement
            c2w_s = cond_canon.copy()
            c2w_s[:, :3, 3] *= scale
            input_c2ws_s = torch.from_numpy(c2w_s).unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                backbone, _, _ = model.backbone(
                    images.to(torch.bfloat16), input_c2ws_s.to(torch.bfloat16),
                    input_intr.to(torch.bfloat16), input_times.to(torch.bfloat16),
                    ot.to(torch.bfloat16), frames_chunk_size=16)
            depth = model.gs_renderer.depth_activation(backbone["depth"].squeeze(2).float())  # (B, F_in, H, W)
            xyz = unproject_depth(depth, input_c2ws_s.float(), input_intr.float())  # (B, F_in, 3, H, W)
            mean_norm = float(xyz.norm(dim=2).mean().item())
            del backbone
            torch.cuda.empty_cache()
            if not (np.isfinite(mean_norm) and mean_norm > 1e-8):
                return rig_scale
            scale = scale / mean_norm
        if not (np.isfinite(scale) and scale > 1e-12):
            return rig_scale
        return float(scale)
    except Exception:
        import traceback
        traceback.print_exc()
        print("_pc_scale failed -> falling back to rig-radius scale", flush=True)
        return rig_scale


def _kubric():
    import bench_ds_kubric4d as m
    return dict(scenes=m.KUBRIC4D_SCENES, load=m.load_kubric4d_scene,
                gt_hw=(384, 576), render_hw=(336, 504), groups=("all",))


def _camxtime():
    import bench_ds_camxtime as m
    return dict(scenes=m.list_test_scenes(), load=m.load_camxtime_scene,
                gt_hw=(512, 512), render_hw=(504, 504), groups=("all",))


def _charge():
    import bench_ds_charge as m
    return dict(scenes=m.SCENES, load=m.load_charge_scene,
                gt_hw=(858, 2048), render_hw=(448, 1064), groups=("all",))


def _n3dv():
    import bench_ds_n3dv as m
    return dict(scenes=m.N3DV_SCENES, load=m.load_n3dv_scene,
                gt_hw=(1014, 1352), render_hw=(504, 672),
                groups=("all",))


def _nvidia():
    import bench_ds_nvidia as m
    return dict(scenes=list(m.NVIDIA_SCENES), load=m.load_nvidia,
                gt_hw=(540, 960), render_hw=(504, 896), groups=("all",))


# Monocular novel-time protocols.
def _adt():
    import bench_ds_adt as m
    _r = os.environ.get("MOVIES_RENDER_HW")
    _hw = tuple(int(x) for x in _r.split("x")) if _r else (504, 504)
    return dict(scenes=list(m.ADT_SCENES), load=m.load_adt_scene,
                gt_hw=(512, 512), render_hw=_hw, groups=("all",))


def _dycheck():
    import bench_ds_dycheck as m
    return dict(scenes=list(m.DYCHECK_SCENES), load=m.load_dycheck_scene,
                gt_hw=(960, 720), render_hw=(896, 672), groups=("all",))


def _tum():
    import bench_ds_tum as m
    return dict(scenes=list(m.TUM_SCENES), load=m.load_tum_scene,
                gt_hw=(480, 640), render_hw=(336, 448), groups=("all",))


DATASETS = {"kubric4d": _kubric, "camxtime": _camxtime, "charge": _charge,
            "n3dv": _n3dv, "nvidia": _nvidia,
            "adt": _adt, "dycheck": _dycheck, "tum": _tum}


TGT_BATCH = int(os.environ.get("MOVIES_TGT_BATCH", "8"))  # targets per backbone call


def run_scene(model, scorer, cfg, scene, out_dir, device):
    d = cfg["load"](scene)
    cond, gt_hw = d["cond"], cfg["gt_hw"]
    render_hw = C.render_hw_override(gt_hw, cfg["render_hw"], 14)
    RH, RW = render_hw
    GT_H, GT_W = gt_hw
    items = C.flatten_items(d["targets"])
    cond_c2ws = np.stack([c["c2w"] for c in cond])
    tgt_c2ws_raw = np.stack([it["c2w"] for _, it in items])
    cond_frames = [c["frame"] for c in cond]
    tgt_frames = [it["frame"] for _, it in items]
    # MoVieS time in [0, 1], normalized over cond+target frames (or cond only).
    fspan = cond_frames if TIME_SPAN == "cond" else (cond_frames + tgt_frames)
    fmin, fmax = min(fspan), max(fspan)
    tscale = float(fmax - fmin) if fmax > fmin else 1.0

    imgs, in_fxfycxcy = [], []
    for c in cond:
        img, K = C.resize_img_K(c["image"], c["K"], RH, RW)
        imgs.append(img); in_fxfycxcy.append(C.norm_intrinsics(K, RH, RW))

    images = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    input_intr = torch.from_numpy(np.stack(in_fxfycxcy)).unsqueeze(0).to(device)
    input_times = torch.from_numpy(
        (np.array(cond_frames, np.float32) - fmin) / tscale).unsqueeze(0).to(device)
    output_times = torch.from_numpy(
        (np.array(tgt_frames, np.float32) - fmin) / tscale).unsqueeze(0).to(device)

    if POSE_SCALE == "baseline":
        base, M = C.fit_norm_4dgt(cond_c2ws, fallback_c2ws=tgt_c2ws_raw)
        cond_norm = C.apply_norm_4dgt(cond_c2ws, base, M)
        tgt_norm = C.apply_norm_4dgt(tgt_c2ws_raw, base, M)
    else:
        canon, rig_scale, radius = C.fit_norm_movies(cond_c2ws, fallback_c2ws=tgt_c2ws_raw)
        if POSE_SCALE == "pc":
            scale = _pc_scale(model, images, cond_c2ws, canon, input_intr,
                              input_times, device, rig_scale)
        else:
            scale = rig_scale
        cond_norm = C.apply_norm_movies(cond_c2ws, canon, scale)
        tgt_norm = C.apply_norm_movies(tgt_c2ws_raw, canon, scale)
    tgt_fxfycxcy = np.stack(
        [C.norm_intrinsics(C.scale_K(it["K"], GT_H, GT_W, RH, RW), RH, RW) for _, it in items])

    input_c2ws = torch.from_numpy(cond_norm).unsqueeze(0).to(device)

    tgt_c2ws_t = torch.from_numpy(tgt_norm).to(device)
    tgt_intr_t = torch.from_numpy(tgt_fxfycxcy).to(device)
    # Process targets in batches to bound memory.
    preds = []
    N = len(items)
    for s in range(0, N, TGT_BATCH):
        e = min(s + TGT_BATCH, N)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone, motions, motion_gaussians = model.backbone(
                images.to(torch.bfloat16), input_c2ws.to(torch.bfloat16),
                input_intr.to(torch.bfloat16), input_times.to(torch.bfloat16),
                output_times[:, s:e].to(torch.bfloat16), frames_chunk_size=16)
        for j in range(e - s):
            dynamic = dict(backbone)
            dynamic["offset"] = motions[:, j, :, :3]
            if motion_gaussians is not None:
                dynamic.update(motion_gaussians[j])
            rc2w = tgt_c2ws_t[s + j:s + j + 1].unsqueeze(0).to(torch.bfloat16)
            rintr = tgt_intr_t[s + j:s + j + 1].unsqueeze(0).to(torch.bfloat16)
            out = model.gs_renderer.render(
                dynamic, input_c2ws.to(torch.bfloat16), input_intr.to(torch.bfloat16),
                rc2w, rintr)["image"]
            preds.append(out[0, 0].float().permute(1, 2, 0).cpu().numpy())
        del backbone, motions, motion_gaussians
        torch.cuda.empty_cache()

    return C.score_and_save(items, preds, scorer, out_dir / scene, scene, render_hw)


def _gpu_free_mib(dev=0):
    return torch.cuda.mem_get_info(dev)[0] // (1024 * 1024)


def wait_for_gpu(min_free, poll=30, max_wait=36000):
    """Wait until at least min_free MiB of GPU memory is free."""
    import time
    t0 = time.time()
    while _gpu_free_mib() < min_free:
        if time.time() - t0 > max_wait:
            print(f"  [mem] proceeding with {_gpu_free_mib()} MiB after {max_wait}s", flush=True)
            return
        print(f"  [mem] {_gpu_free_mib()} MiB free (<{min_free}); wait {poll}s ...", flush=True)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--scenes", nargs="+", default=None)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]()
    scenes = args.scenes or list(cfg["scenes"])
    out_dir = OUT_ROOT / args.dataset / "movies"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    _o = opt_dict["movies"]
    _mm = os.environ.get("MOVIES_MASK_MOTION")
    if _mm:
        _o.mask_by_motion = float(_mm); _o.mask_by_motion_opacity = float(_mm)
        print(f"[movies] mask_by_motion overridden -> {_o.mask_by_motion}", flush=True)
    model = SplatRecon(_o, load_lpips=False)
    model.load_state_dict(load_file(str(CKPT)), strict=True)
    model = model.to(device).eval()
    scorer = Scorer(device="cuda")

    MIN_FREE = int(os.environ.get("BENCH_MIN_FREE", "0"))
    per_scene = {}
    for sc in scenes:
        for attempt in range(3):
            try:
                if MIN_FREE:
                    wait_for_gpu(MIN_FREE)
                with torch.inference_mode():
                    per_scene[sc] = run_scene(model, scorer, cfg, sc, out_dir, device)
                msg = " | ".join(
                    f"{g} PSNR {per_scene[sc][g]['psnr']:.3f} dyn {per_scene[sc][g].get('dyn_psnr', float('nan')):.3f}"
                    for g in cfg["groups"])
                print(f"{sc}: {msg}", flush=True)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                import time
                print(f"{sc}: OOM, retry in 30s ({attempt + 1}/3)", flush=True)
                time.sleep(30)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"{sc}: FAILED {e}", flush=True)
                break
        torch.cuda.empty_cache()

    if per_scene:
        pose_desc = {
            "baseline": "canon cam0 + /max cond baseline == 1 (reference eval.py)",
            "rig": "canon cam0 + /median rig radius",
            "pc": "canon cam0 + /VGGT point-cloud magnitude",
        }.get(POSE_SCALE, POSE_SCALE)
        time_desc = ("(frame - min_cond_frame) / cond_frame_span in [0,1]"
                     if TIME_SPAN == "cond" else
                     "frame normalized over cond+target union in [0,1] (reference)")
        framing = {"render_hw": list(C.render_hw_override(cfg["gt_hw"], cfg["render_hw"], 14)), "gt_hw": list(cfg["gt_hw"]),
                   "input": "full native-aspect, no crop (/14)",
                   "score_res": "native GT, pred bilinear-upsampled", "lpips": "vgg",
                   "pose_norm": pose_desc + " (fit cond, applied to targets)",
                   "time_token": time_desc}
        summary = C.write_summary(out_dir, "MoVieS", args.dataset, per_scene, framing, cfg["groups"])
        print("\nOVERALL", {k: round(summary["overall"][k], 4) for k in C.METRIC_KEYS})
        for g in cfg["groups"]:
            print(g.upper(), {k: round(summary[g][k], 4) for k in C.METRIC_KEYS})
        print("saved ->", out_dir / "summary.json")


if __name__ == "__main__":
    main()
