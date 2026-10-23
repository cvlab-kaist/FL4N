#!/usr/bin/env python3
"""4DGT runner: protocol loaders -> 4DGT encode/render -> protocol scorer."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_B = Path(__file__).resolve().parent.parent       # benchmark/
for _d in ("common", "datasets", "runners"):
    sys.path.insert(0, str(_B / _d))
from bench_paths import BASELINES, OUT, weights  # noqa: E402

REPO = BASELINES / "4DGT"
sys.path.insert(3, str(REPO))
from tlod.demo import FourDGTDemo  # noqa: E402
from tlod.data_loader.utils import pack_c2ws_to_cameras  # noqa: E402
from tlod.easyvolcap.utils.cam_utils import average_c2ws  # noqa: E402
from protocol_score import Scorer  # noqa: E402
import bench_proto_common as C  # noqa: E402


# Pose normalization: canonicalize to cam0, scale by the cond baseline.
def fit_norm(cond_c2ws, fallback_c2ws=None):
    if os.environ.get("POSE_MODE", "scale") == "scale":
        return C.fit_norm_4dgt(cond_c2ws, fallback_c2ws=fallback_c2ws)
    M = np.linalg.inv(cond_c2ws[0].astype(np.float64))
    return 1.0, M


def apply_norm(c2ws, base, M):
    if os.environ.get("POSE_MODE", "scale") == "scale":
        return C.apply_norm_4dgt(c2ws, base, M)
    n = c2ws.astype(np.float64).copy()
    n[:, :3, 3] /= base
    return (M[None] @ n).astype(np.float32)

CKPT = str(weights("4dgt", "4dgt_full.pth"))
CONFIG = str(REPO / "configs/models/tlod-l3.py")
TSPAN = float(os.environ.get("BENCH_TSPAN", "0.25"))  # cond temporal span, seconds
FPS_LEGACY = os.environ.get("BENCH_FPS")
OUT_ROOT = OUT


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
    return dict(scenes=list(m.ADT_SCENES), load=m.load_adt_scene,
                gt_hw=(512, 512), render_hw=(504, 504), groups=("all",))


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


def to_model(img):
    return np.transpose(np.clip(img, 0, 1) * 2 - 1, (2, 0, 1)).astype(np.float32)


def to_img(chw):
    return np.clip((np.transpose(chw, (1, 2, 0)) + 1) / 2, 0, 1)


def run_scene(demo, scorer, cfg, scene, out_dir):
    d = cfg["load"](scene)
    cond, gt_hw = d["cond"], cfg["gt_hw"]
    render_hw = C.render_hw_override(gt_hw, cfg["render_hw"], 28)
    RH, RW = render_hw
    GT_H, GT_W = gt_hw
    items = C.flatten_items(d["targets"])
    cond_c2ws = np.stack([c["c2w"] for c in cond])
    tgt_c2ws_raw = np.stack([it["c2w"] for _, it in items])
    base, M = fit_norm(cond_c2ws, fallback_c2ws=tgt_c2ws_raw)
    cond_norm = apply_norm(cond_c2ws, base, M)

    model_imgs, in_Ks = [], []
    for c in cond:
        img, K = C.resize_img_K(c["image"], c["K"], RH, RW)
        model_imgs.append(to_model(img)); in_Ks.append(K)
    # Time token in seconds: origin at the earliest cond frame, span rescaled to TSPAN.
    fmin = min(c["frame"] for c in cond)
    fmax = max(c["frame"] for c in cond)
    if FPS_LEGACY is not None:
        tscale = 1.0 / float(FPS_LEGACY)
    else:
        span = fmax - fmin
        tscale = (TSPAN / span) if span > 0 else 0.0  # static cond -> all times 0
    in_times = [(c["frame"] - fmin) * tscale for c in cond]
    # 4DGT encodes frames in temporal windows of 4; pad to a multiple of 4.
    npad = (-len(model_imgs)) % 4
    if npad:
        cond_norm = np.concatenate([cond_norm, np.repeat(cond_norm[-1:], npad, 0)], 0)
        model_imgs += [model_imgs[-1]] * npad
        in_Ks += [in_Ks[-1]] * npad
        in_times += [in_times[-1]] * npad
    NIN = len(model_imgs)
    in_cams = pack_c2ws_to_cameras(cond_norm, np.asarray(in_Ks),
                                   np.full(NIN, RH, np.int32),
                                   np.full(NIN, RW, np.int32)).astype(np.float32)
    gd = demo.encode(torch.from_numpy(np.stack(model_imgs))[None],
                     torch.from_numpy(in_cams)[None],
                     torch.from_numpy(np.asarray(in_times, np.float32))[None])

    tgt_c2ws = apply_norm(tgt_c2ws_raw, base, M)
    tgt_Ks = [C.scale_K(it["K"], GT_H, GT_W, RH, RW) for _, it in items]
    tgt_times = [(it["frame"] - fmin) * tscale for _, it in items]
    # pad the target batch to a multiple of 4 as well
    n_real = len(items)
    tpad = (-n_real) % 4
    if tpad:
        tgt_c2ws = np.concatenate([tgt_c2ws, np.repeat(tgt_c2ws[-1:], tpad, 0)], 0)
        tgt_Ks = tgt_Ks + [tgt_Ks[-1]] * tpad
        tgt_times = tgt_times + [tgt_times[-1]] * tpad
    NT = n_real + tpad
    tgt_cams = pack_c2ws_to_cameras(tgt_c2ws, np.asarray(tgt_Ks),
                                    np.full(NT, RH, np.int32),
                                    np.full(NT, RW, np.int32)).astype(np.float32)

    tc = torch.from_numpy(tgt_cams)[None]
    tt = torch.from_numpy(np.asarray(tgt_times, np.float32))[None]
    preds = []
    for s in range(0, NT, 8):
        e = min(s + 8, NT)
        out = demo.render_batch(gd, tc[:, s:e], tt[:, s:e], render_mode="RGB+ED")
        preds.extend(to_img(x) for x in out["rgb"][0].float().cpu().numpy())
        torch.cuda.empty_cache()
    preds = preds[:n_real]  # drop padding

    return C.score_and_save(items, preds, scorer, out_dir / scene, scene, render_hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--scenes", nargs="+", default=None)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]()
    scenes = args.scenes or list(cfg["scenes"])
    out_dir = OUT_ROOT / args.dataset / "4dgt"
    out_dir.mkdir(parents=True, exist_ok=True)

    # FourDGTDemo writes scratch subdirs under output_dir.
    _demo_scratch = str(OUT_ROOT / "_4dgt_scratch")
    Path(_demo_scratch).mkdir(parents=True, exist_ok=True)
    demo = FourDGTDemo(config_path=CONFIG, checkpoint_path=CKPT, device="cuda",
                       output_dir=_demo_scratch, fp32=False)
    scorer = Scorer(device="cuda")

    per_scene = {}
    for sc in scenes:
        try:
            per_scene[sc] = run_scene(demo, scorer, cfg, sc, out_dir)
            msg = " | ".join(
                f"{g} PSNR {per_scene[sc][g]['psnr']:.3f} dyn {per_scene[sc][g].get('dyn_psnr', float('nan')):.3f}"
                for g in cfg["groups"])
            print(f"{sc}: {msg}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{sc}: FAILED {e}", flush=True)
        torch.cuda.empty_cache()

    if per_scene:
        framing = {"render_hw": list(C.render_hw_override(cfg["gt_hw"], cfg["render_hw"], 28)), "gt_hw": list(cfg["gt_hw"]),
                   "input": "full native-aspect, no crop (/56)",
                   "score_res": "native GT, pred bilinear-upsampled", "lpips": "vgg",
                   "pose_norm": "4DGT canon cam0 + /max cond baseline (fit cond, applied to targets)",
                   "time_token": f"cond span rescaled to {TSPAN}s (fits +/-0.5s dxyzt budget), relative spacing preserved, shared cond+target"}
        summary = C.write_summary(out_dir, "4DGT", args.dataset, per_scene, framing, cfg["groups"])
        print("\nOVERALL", {k: round(summary["overall"][k], 4) for k in C.METRIC_KEYS})
        for g in cfg["groups"]:
            print(g.upper(), {k: round(summary[g][k], 4) for k in C.METRIC_KEYS})
        print("saved ->", out_dir / "summary.json")


if __name__ == "__main__":
    main()
