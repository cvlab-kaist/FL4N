#!/usr/bin/env python3
"""NeoVerse runner (pose-free reconstructor, target-view pose TTO): protocol loaders -> render -> protocol scorer."""
from __future__ import annotations

import argparse
import os
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_B = Path(__file__).resolve().parent.parent       # benchmark/
for _d in ("common", "datasets", "runners"):
    sys.path.insert(0, str(_B / _d))
from bench_paths import BASELINES, OUT, weights  # noqa: E402

NEOVERSE = BASELINES / "NeoVerse"
sys.path.insert(3, str(NEOVERSE))
from eval_nvidia_sparse3 import NeoVerseReconstructor  # noqa: E402
from protocol_score import Scorer  # noqa: E402
import bench_ds_kubric4d as k4  # noqa: E402
import bench_proto_common as C  # noqa: E402
import bench_ds_camxtime as cx  # noqa: E402
import bench_ds_charge as ch  # noqa: E402
import bench_ds_n3dv as n3  # noqa: E402
import bench_ds_nvidia as nvt2  # noqa: E402

CKPT = weights("neoverse", "reconstructor.ckpt")
OUT_BASE = OUT

# NEO_FIX: scale-to-cover + center-crop to the NeoVerse operating point; NEO_FIX=res: native aspect.
_NEO_FIX_RAW = os.environ.get("NEO_FIX", "")
NEO_FIX_RES = (_NEO_FIX_RAW == "res")
NEO_FIX = bool(_NEO_FIX_RAW) and not NEO_FIX_RES  # crop variant
FIX_HW = (336, 560)  # (H, W), both /14
FIX_LONG = 560  # long side for NEO_FIX=res

# Reference model-side handling (context-only Gaussians + GT target poses).
NEO_REF = os.environ.get("NEO_REF", "1") != "0"
# GT context poses -> NeoVerse predicted context frame alignment.
NEO_ALIGN = os.environ.get("NEO_ALIGN", "scale").lower()
NEO_TTO_TARGET = os.environ.get("NEO_TTO_TARGET", "0") != "0"
NEO_TTO_TGT_CHUNK = int(os.environ.get("NEO_TTO_TGT_CHUNK", "8"))
NEO_NORM_POSE = os.environ.get("NEO_NORM_POSE", "1") != "0"
NEO_NORM_SCALE = os.environ.get("NEO_NORM_SCALE", "1") != "0"
# Test-time pose optimisation (opt-in).
NEO_TTO = os.environ.get("NEO_TTO", "") not in ("", "0")
NEO_TTO_STEPS = int(os.environ.get("NEO_TTO_STEPS", "300"))
NEO_TTO_ROT_LR = float(os.environ.get("NEO_TTO_ROT_LR", "1e-3"))
NEO_TTO_TRANS_LR = float(os.environ.get("NEO_TTO_TRANS_LR", "1e-3"))
# Static-cond (zero-baseline) fallback.
NEO_STATIC_COND = os.environ.get("NEO_STATIC_COND", "1") not in ("", "0")
NEO_STATIC_COND_S = float(os.environ.get("NEO_STATIC_COND_S", "1.0"))


def res_match_hw(native_hw, long_target=FIX_LONG, patch=14):
    """Native-aspect render size with long side ~= long_target, both /patch."""
    nh, nw = native_hw
    scale = long_target / float(max(nh, nw))
    H = max(patch, int(round(nh * scale / patch)) * patch)
    W = max(patch, int(round(nw * scale / patch)) * patch)
    return H, W

# Render sizes: divisible by 14 (NeoVerse patch), aspect matched to native GT.
DATASETS = {
    "kubric4d": dict(render_hw=(336, 504), groups=("all",),
                     native_hw=(384, 576), has_dyn=True, min_free=8000),
    "camxtime": dict(render_hw=(504, 504), groups=("all",),
                     native_hw=(512, 512), has_dyn=False, min_free=12000),
    "charge":   dict(render_hw=(252, 602), groups=("all",),
                     native_hw=(858, 2048), has_dyn=True, min_free=12000),
    "n3dv":     dict(render_hw=(504, 672), groups=("all",),
                     native_hw=(1014, 1352), has_dyn=False, min_free=14000),
    "nvidia":   dict(render_hw=(504, 896), groups=("all",),
                     native_hw=(540, 960), has_dyn=False, min_free=10000),
    # Monocular novel-time protocols.
    "adt":      dict(render_hw=(504, 504), groups=("all",),
                     native_hw=(512, 512), has_dyn=False, min_free=12000),
    "dycheck":  dict(render_hw=(896, 672), groups=("all",),
                     native_hw=(960, 720), has_dyn=False, min_free=12000),
    "tum":      dict(render_hw=(462, 616), groups=("all",),
                     native_hw=(480, 640), has_dyn=False, min_free=12000),
}


def load_scene(dataset, scene):
    if dataset == "kubric4d":
        return k4.load_kubric4d_scene(scene)
    if dataset == "camxtime":
        return cx.load_camxtime_scene(scene)
    if dataset == "charge":
        return ch.load_charge_scene(scene)
    if dataset == "n3dv":
        return n3.load_n3dv_scene(scene)
    if dataset == "nvidia":
        return nvt2.load_nvidia(scene)
    if dataset == "adt":
        import bench_ds_adt as ad; return ad.load_adt_scene(scene)
    if dataset == "dycheck":
        import bench_ds_dycheck as dc; return dc.load_dycheck_scene(scene)
    if dataset == "tum":
        import bench_ds_tum as tm; return tm.load_tum_scene(scene)
    raise ValueError(dataset)


def scene_list(dataset):
    if dataset == "kubric4d":
        return list(k4.KUBRIC4D_SCENES)
    if dataset == "camxtime":
        return list(cx.list_test_scenes())
    if dataset == "charge":
        return list(ch.SCENES)
    if dataset == "n3dv":
        return list(n3.N3DV_SCENES)
    if dataset == "nvidia":
        return list(nvt2.NVIDIA_SCENES)
    if dataset == "adt":
        import bench_ds_adt as ad; return list(ad.ADT_SCENES)
    if dataset == "dycheck":
        import bench_ds_dycheck as dc; return list(dc.DYCHECK_SCENES)
    if dataset == "tum":
        import bench_ds_tum as tm; return list(tm.TUM_SCENES)
    raise ValueError(dataset)


def resize_to_tensor(img_hwc01, H, W):
    """numpy HxWx3 [0,1] -> torch (3,H,W) [0,1] at (H,W) via LANCZOS."""
    a = np.clip(np.asarray(img_hwc01, np.float32), 0, 1)
    pil = Image.fromarray((a * 255).astype(np.uint8))
    if pil.size != (W, H):
        pil = pil.resize((W, H), Image.Resampling.LANCZOS)
    arr = np.asarray(pil, np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def center_crop_to_tensor(img_hwc01, H, W):
    """Scale-to-cover + center-crop numpy HxWx3 [0,1] -> torch (3,H,W) [0,1]."""
    a = np.clip(np.asarray(img_hwc01, np.float32), 0, 1)
    pil = Image.fromarray((a * 255).astype(np.uint8))
    w, h = pil.size  # PIL size is (W, H)
    scale = max(W / w, H / h)
    ow, oh = int(w * scale), int(h * scale)
    pil = pil.resize((ow, oh), Image.Resampling.LANCZOS)
    left, top = (ow - W) // 2, (oh - H) // 2
    pil = pil.crop((left, top, left + W, top + H))
    arr = np.asarray(pil, np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def center_crop_np(img_hwc01, H, W):
    """Same scale-to-cover + center-crop, returned as numpy HxWx3 [0,1] (GT for scoring)."""
    return center_crop_to_tensor(img_hwc01, H, W).permute(1, 2, 0).numpy()


def center_crop_mask(mask_hw, H, W):
    """Scale-to-cover + center-crop a bool HxW mask (NEAREST) to (H,W)."""
    if mask_hw is None:
        return None
    m = (np.asarray(mask_hw).astype(np.uint8) * 255)
    pil = Image.fromarray(m)
    w, h = pil.size
    scale = max(W / w, H / h)
    ow, oh = int(w * scale), int(h * scale)
    pil = pil.resize((ow, oh), Image.Resampling.NEAREST)
    left, top = (ow - W) // 2, (oh - H) // 2
    pil = pil.crop((left, top, left + W, top + H))
    return np.asarray(pil) > 127


def to_hwc(pred_chw):
    return pred_chw.permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()


def gpu_free_mib():
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 ** 2)


def wait_for_gpu(min_free_mib, timeout_s=1800, poll_s=30):
    _ov = os.environ.get("BENCH_MIN_FREE")
    if _ov:
        min_free_mib = int(_ov)
    t0 = time.time()
    while True:
        free = gpu_free_mib()
        if free >= min_free_mib:
            return free
        if time.time() - t0 > timeout_s:
            print(f"  [mem] proceeding with only {free:.0f} MiB free after {timeout_s}s", flush=True)
            return free
        print(f"  [mem] {free:.0f} MiB < {min_free_mib} needed; wait {poll_s}s ...", flush=True)
        time.sleep(poll_s)


def logical_time_fn(cond):
    """Map a target frame index to the logical cond-time index 0..7 (nearest cond frame)."""
    cond_frames = np.array([int(c["frame"]) for c in cond], dtype=np.float64)

    def f(frame):
        return int(np.argmin(np.abs(cond_frames - float(frame))))
    return f


def agg(rows, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and r[k] is not None and np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else None
    return out


# Reference model-side handling (adapted from NeoVerse eval_iphone.py).
def _gt_c2w_K(items, native_hw, H, W):
    """Stack loader GT c2w [N,4,4] (OpenCV) and K [N,3,3] scaled native->render."""
    nh, nw = native_hw
    sx, sy = W / float(nw), H / float(nh)
    c2ws, Ks = [], []
    for it in items:
        c2ws.append(np.asarray(it["c2w"], np.float64))
        K = np.asarray(it["K"], np.float64).copy()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        Ks.append(K)
    c2w = torch.from_numpy(np.stack(c2ws)).float()
    Kt = torch.from_numpy(np.stack(Ks)).float()
    return c2w, Kt


def _normalize_gt_poses(ctx_c2w, tgt_c2w, do_scale, do_pose):
    """Scale by the max context baseline, then put context cam0 at identity (context + targets)."""
    ctx = ctx_c2w.clone()
    tgt = tgt_c2w.clone()
    if do_scale:
        ref_pos = ctx[0, :3, 3]
        diffs = ctx[1:, :3, 3] - ref_pos
        max_baseline = diffs.norm(dim=1).max() if ctx.shape[0] > 1 else torch.tensor(0.0)
        if os.environ.get("BENCH_NORM_SPAN_ALL") == "1":
            # scale on the cond+target span
            span = torch.cat([ctx[:, :3, 3], tgt[:, :3, 3]], 0).sub(ref_pos).norm(dim=1).max().clamp(min=1e-6)
            ctx[:, :3, 3] /= span
            tgt[:, :3, 3] /= span
        elif max_baseline > 1e-6:
            ctx[:, :3, 3] /= max_baseline
            tgt[:, :3, 3] /= max_baseline
        elif NEO_STATIC_COND:
            span = torch.cat([ctx[:, :3, 3], tgt[:, :3, 3]], 0).sub(ref_pos).norm(dim=1).max().clamp(min=1e-6)
            ctx[:, :3, 3] /= span
            tgt[:, :3, 3] /= span
    if do_pose:
        ref_w2c = torch.linalg.inv(ctx[0])
        ctx = ref_w2c @ ctx
        tgt = ref_w2c @ tgt
    return ctx, tgt


def align_scale_only(c2w_src, c2w_dst):
    """Scale-only alignment (from NeoVerse eval_iphone.py). dst ~= s*(src-mu_src)+mu_dst."""
    pos_src = c2w_src[:, :3, 3]
    pos_dst = c2w_dst[:, :3, 3]
    mu_src = pos_src.mean(dim=0)
    mu_dst = pos_dst.mean(dim=0)
    spread_src = (pos_src - mu_src).norm(dim=1).mean().clamp(min=1e-8)
    spread_dst = (pos_dst - mu_dst).norm(dim=1).mean().clamp(min=1e-8)
    s = spread_dst / spread_src
    t_off = mu_dst - s * mu_src
    return s, t_off


def apply_scale_alignment(s, t_off, c2w):
    c2w_new = c2w.clone()
    c2w_new[:, :3, 3] = s * c2w[:, :3, 3] + t_off
    return c2w_new


def _umeyama(src, dst):
    """Umeyama similarity (from NeoVerse eval_iphone.py). dst ~= s*R@src + t."""
    n = src.shape[0]
    mu_src = src.mean(dim=0)
    mu_dst = dst.mean(dim=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = (src_c ** 2).sum() / n
    Hm = (dst_c.T @ src_c) / n
    U, D, Vt = torch.linalg.svd(Hm)
    S = torch.eye(3, device=src.device, dtype=src.dtype)
    if torch.det(U) * torch.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * S.diag()).sum() / var_src
    T = torch.eye(4, device=src.device, dtype=src.dtype)
    T[:3, :3] = s * R
    T[:3, 3] = mu_dst - s * (R @ mu_src)
    return s, T


def align_poses(ctx_gt_c2w, ctx_tgt_gt_c2w, neo_ctx_c2w, mode):
    """Return (ctx_aligned, tgt_aligned, scale) mapping GT poses -> NeoVerse frame."""
    dev = neo_ctx_c2w.device
    ctx = ctx_gt_c2w.to(dev).float()
    tgt = ctx_tgt_gt_c2w.to(dev).float()
    neo = neo_ctx_c2w.to(dev).float()
    if mode == "umeyama":
        s, T = _umeyama(ctx[:, :3, 3], neo[:, :3, 3])
        if torch.isnan(s):
            return ctx, tgt, torch.tensor(1.0)
        return (T @ ctx), (T @ tgt), s
    if NEO_STATIC_COND and (ctx[:, :3, 3] - ctx[:, :3, 3].mean(0)).norm(dim=1).mean() < 1e-6:
        s = torch.tensor(NEO_STATIC_COND_S, device=dev)
        t_off = neo[:, :3, 3].mean(0) - s * ctx[:, :3, 3].mean(0)
        return apply_scale_alignment(s, t_off, ctx), apply_scale_alignment(s, t_off, tgt), s
    s, t_off = align_scale_only(ctx, neo)
    return apply_scale_alignment(s, t_off, ctx), apply_scale_alignment(s, t_off, tgt), s


@torch.inference_mode()
def neo_encode_context(recon, ctx_imgs01, ctx_ts, use_motion):
    """Context-only reconstruction."""
    imgs = ctx_imgs01
    N = imgs.shape[0]
    dev = recon.device
    # NEO_STATIC: encode context as static frames at timestamp 0.
    static = os.environ.get("NEO_STATIC", "1") not in ("", "0")
    if static:
        is_static = torch.ones(1, N, dtype=torch.bool, device=dev)
        ts = torch.zeros(1, N, dtype=torch.int64, device=dev)
        use_motion = False
    else:
        is_static = torch.zeros(1, N, dtype=torch.bool, device=dev)
        ts = ctx_ts.to(torch.int64)[None].to(dev)
    views = {
        "img": imgs[None].to(dev, non_blocking=True),
        "is_target": torch.zeros(1, N, dtype=torch.bool, device=dev),
        "is_static": is_static,
        "timestamp": ts,
    }
    with torch.amp.autocast("cuda", dtype=recon.dtype):
        preds = recon.model(views, is_inference=True, use_motion=use_motion)
    return {
        "splats": preds["splats"],
        "c2w": preds["rendered_extrinsics"][0].float(),
        "intrinsics": preds["rendered_intrinsics"][0].float(),
        "timestamps": preds["rendered_timestamps"][0],
    }


# ---- SE3 helpers + TTO (adapted from NeoVerse eval_iphone.py) ---- #
def _skew(x):
    o = torch.zeros(3, 3, device=x.device, dtype=x.dtype)
    o[0, 1] = -x[2]; o[0, 2] = x[1]
    o[1, 0] = x[2]; o[1, 2] = -x[0]
    o[2, 0] = -x[1]; o[2, 1] = x[0]
    return o


def _so3_exp(theta):
    W = _skew(theta)
    angle = theta.norm()
    I = torch.eye(3, device=theta.device, dtype=theta.dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W @ W
    return I + (torch.sin(angle) / angle) * W + ((1 - torch.cos(angle)) / angle ** 2) * (W @ W)


def _v_mat(theta):
    W = _skew(theta)
    W2 = W @ W
    angle = theta.norm()
    I = torch.eye(3, device=theta.device, dtype=theta.dtype)
    if angle < 1e-5:
        return I + 0.5 * W + (1.0 / 6.0) * W2
    return I + ((1 - torch.cos(angle)) / angle ** 2) * W + ((angle - torch.sin(angle)) / angle ** 3) * W2


def _se3_exp(tau):
    rho, theta = tau[:3], tau[3:]
    T = torch.eye(4, device=tau.device, dtype=tau.dtype)
    T[:3, :3] = _so3_exp(theta)
    T[:3, 3] = _v_mat(theta) @ rho
    return T


def _apply_se3_delta(w2c, rot_d, trans_d):
    out = []
    for i in range(w2c.shape[0]):
        tau = torch.cat([trans_d[i], rot_d[i]], dim=-1)
        out.append(_se3_exp(tau) @ w2c[i])
    return torch.stack(out, dim=0)


def neo_optimize_pose(recon, enc, gt_images, c2w_init, K, ts, H, W,
                      rot_lr, trans_lr, num_steps):
    """Freeze Gaussians and optimise camera poses to match GT."""
    dev = recon.device
    gt = gt_images.to(dev)
    K_dev = K.to(dev).float()
    ts_dev = ts.to(dev)
    M = c2w_init.shape[0]
    c2w_cur = c2w_init.to(dev).float()
    w2c_cur = torch.linalg.inv(c2w_cur)
    rot_d = torch.nn.Parameter(torch.zeros(M, 3, device=dev))
    trans_d = torch.nn.Parameter(torch.zeros(M, 3, device=dev))
    opt = torch.optim.Adam([{"params": [rot_d], "lr": rot_lr},
                            {"params": [trans_d], "lr": trans_lr}])
    for step in range(num_steps):
        opt.zero_grad()
        new_w2c = _apply_se3_delta(w2c_cur, rot_d, trans_d)
        with torch.amp.autocast("cuda", dtype=recon.dtype):
            rgb, _, _ = recon.model.gs_renderer.rasterizer.forward(
                enc["splats"], render_viewmats=[new_w2c], render_Ks=[K_dev],
                render_timestamps=[ts_dev], sh_degree=0, width=W, height=H)
        pred = rgb[0].permute(0, 3, 1, 2).clamp(0, 1)
        loss = torch.nn.functional.mse_loss(pred, gt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            w2c_cur = _apply_se3_delta(w2c_cur, rot_d, trans_d).detach()
            rot_d.data.zero_(); trans_d.data.zero_()
    return torch.linalg.inv(w2c_cur)


def _run_scene_ref(recon, scorer, dataset, scene, cfg, out_dir, H, W, prep):
    """Reference model-side handling: context-only Gaussians + GT target poses."""
    native_hw = cfg["native_hw"]
    groups = cfg["groups"]
    data = load_scene(dataset, scene)
    cond, targets = data["cond"], data["targets"]
    log_t = logical_time_fn(cond)

    ctx = torch.stack([prep(c["image"], H, W) for c in cond])   # [8,3,H,W] [0,1]
    ctx_ts = torch.arange(len(cond), dtype=torch.int64)

    tgt_items, group_of = [], []
    for g in groups:
        for t in targets.get(g, []):
            tgt_items.append(t)
            group_of.append(g)
    tgt_ts = torch.tensor([log_t(t["frame"]) for t in tgt_items], dtype=torch.int64)

    # GT poses (loader) -> normalize -> align to NeoVerse predicted context frame.
    ctx_c2w_gt, _ = _gt_c2w_K(cond, native_hw, H, W)
    tgt_c2w_gt, tgt_K = _gt_c2w_K(tgt_items, native_hw, H, W)
    ctx_c2w_n, tgt_c2w_n = _normalize_gt_poses(
        ctx_c2w_gt, tgt_c2w_gt, NEO_NORM_SCALE, NEO_NORM_POSE)

    # ---- context-only encode (predicts context cameras + Gaussians) ----
    for attempt in range(6):
        try:
            wait_for_gpu(cfg["min_free"])
            torch.cuda.reset_peak_memory_stats()
            enc = neo_encode_context(recon, ctx, ctx_ts,
                                     use_motion=(os.environ.get("NEO_USE_MOTION", "1") != "0"))
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [OOM encode attempt {attempt}] retry in 30s", flush=True)
            time.sleep(30)
    else:
        raise RuntimeError(f"{scene}: encode OOM after retries")

    neo_ctx_c2w = enc["c2w"].detach()
    ctx_c2w_al, tgt_c2w_al, s_align = align_poses(
        ctx_c2w_n, tgt_c2w_n, neo_ctx_c2w, NEO_ALIGN)
    # NEO_POSE_SCALE_MULT: scale the ALIGNED cameras about the NeoVerse frame origin.
    _neo_mult = float(os.environ.get("NEO_POSE_SCALE_MULT", "1.0"))
    if _neo_mult != 1.0:
        ctx_c2w_al = ctx_c2w_al.clone(); tgt_c2w_al = tgt_c2w_al.clone()
        ctx_c2w_al[:, :3, 3] *= _neo_mult
        tgt_c2w_al[:, :3, 3] *= _neo_mult
    print(f"  [ref] align={NEO_ALIGN} s={float(s_align):.4f} "
          f"norm_pose={NEO_NORM_POSE} norm_scale={NEO_NORM_SCALE} tto={NEO_TTO}",
          flush=True)

    # Optional TTO: refine on context views, propagate delta to targets.
    if NEO_TTO:
        ctx_ts_dev = ctx_ts
        c2w_opt = neo_optimize_pose(
            recon, enc, gt_images=ctx, c2w_init=ctx_c2w_al,
            K=_gt_c2w_K(cond, native_hw, H, W)[1], ts=ctx_ts_dev, H=H, W=W,
            rot_lr=NEO_TTO_ROT_LR, trans_lr=NEO_TTO_TRANS_LR, num_steps=NEO_TTO_STEPS)
        ref_delta = c2w_opt[0] @ torch.linalg.inv(ctx_c2w_al[0])
        tgt_c2w_al = ref_delta @ tgt_c2w_al

    # NEO_TTO_TARGET: refine each target pose against its own GT image.
    if NEO_TTO and NEO_TTO_TARGET:
        _tgt_gt = torch.stack([prep(t["gt"], H, W) for t in tgt_items])
        _opt_chunks = []
        for _s in range(0, len(tgt_items), NEO_TTO_TGT_CHUNK):
            _e = min(_s + NEO_TTO_TGT_CHUNK, len(tgt_items))
            _opt_chunks.append(neo_optimize_pose(
                recon, enc, gt_images=_tgt_gt[_s:_e], c2w_init=tgt_c2w_al[_s:_e],
                K=tgt_K[_s:_e], ts=tgt_ts[_s:_e], H=H, W=W,
                rot_lr=NEO_TTO_ROT_LR, trans_lr=NEO_TTO_TRANS_LR,
                num_steps=NEO_TTO_STEPS).detach().cpu())
        tgt_c2w_al = torch.cat(_opt_chunks, 0).to(tgt_c2w_al.device).float()
        print(f"  [ref] ORACLE target-image TTO applied to {len(tgt_items)} targets "
              f"({NEO_TTO_STEPS} steps, chunk {NEO_TTO_TGT_CHUNK})", flush=True)

    sdir = out_dir / scene
    (sdir / "pred").mkdir(parents=True, exist_ok=True)

    rows = []
    preds_small = {}
    n = len(tgt_items)
    vis_idx = np.linspace(0, n - 1, min(8, n)).round().astype(int).tolist()
    chunk = 8
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        for attempt in range(6):
            try:
                pred = recon.render(
                    enc,
                    c2w_target=tgt_c2w_al[s:e],
                    k_target=tgt_K[s:e],
                    timestamps_target=tgt_ts[s:e],
                    height=H, width=W)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [OOM render {s}:{e} attempt {attempt}] retry in 30s", flush=True)
                time.sleep(30)
        else:
            raise RuntimeError(f"{scene}: render OOM {s}:{e}")

        for j in range(e - s):
            idx = s + j
            item = tgt_items[idx]
            g = group_of[idx]
            phwc = to_hwc(pred[j])
            covis = item.get("covis_mask", None)
            if NEO_FIX:
                gt = center_crop_np(item["gt"], H, W)
                dyn = center_crop_mask(item.get("dyn_mask", None), H, W) if cfg["has_dyn"] else None
                if covis is not None:
                    covis = center_crop_mask(covis, H, W)
            else:
                gt = np.clip(np.asarray(item["gt"], np.float32), 0, 1)
                dyn = item.get("dyn_mask", None) if cfg["has_dyn"] else None
            m = scorer.score(phwc, gt, dyn_mask=dyn, covis_mask=covis)
            row = {"group": g, "idx": idx, "frame": item.get("frame"),
                   "psnr": m["psnr"], "ssim": m["ssim"], "lpips": m["lpips"]}
            if "offset" in item:
                row["offset"] = item["offset"]
            if "dyn_psnr" in m:
                row["dyn_psnr"] = m["dyn_psnr"]
            rows.append(row)
            # save pred at native GT resolution
            _tag = f"v{C._vtag(item)}_f{int(item['frame']):02d}"
            _gh, _gw = gt.shape[:2]
            _out = np.clip(phwc, 0, 1)
            if _out.shape[:2] != (_gh, _gw):
                _out = np.clip(cv2.resize(_out, (_gw, _gh), interpolation=cv2.INTER_LINEAR), 0, 1)
            Image.fromarray((_out * 255 + 0.5).astype(np.uint8)).save(
                sdir / "pred" / f"{_tag}.png")
            if idx in vis_idx:
                preds_small[idx] = phwc
        del pred
        torch.cuda.empty_cache()

    metric_keys = ["psnr", "ssim", "lpips"] + (["dyn_psnr"] if cfg["has_dyn"] else [])
    scene_groups = {}
    for g in groups:
        grows = [r for r in rows if r["group"] == g]
        if grows:
            scene_groups[g] = agg(grows, metric_keys)
    scene_overall = agg(rows, metric_keys)

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    metrics = {"scene": scene, "dataset": dataset, "render_hw": [H, W],
               "n": len(rows), "peak_gib": peak,
               "model_side": f"ref(align={NEO_ALIGN},tto_target={NEO_TTO_TARGET},norm_pose={NEO_NORM_POSE},"
                             f"norm_scale={NEO_NORM_SCALE},tto={NEO_TTO})",
               "overall": scene_overall, **scene_groups, "rows": rows}
    json.dump(metrics, open(sdir / "metrics.json", "w"), indent=2)

    cols = []
    for i in vis_idx:
        gt = np.clip(np.asarray(tgt_items[i]["gt"], np.float32), 0, 1)   # native GT, as scored
        _gh, _gw = gt.shape[:2]
        pr = np.clip(cv2.resize(np.clip(preds_small[i], 0, 1).astype(np.float32), (_gw, _gh),
                                interpolation=cv2.INTER_LINEAR), 0, 1)
        er = np.clip(np.abs(gt - pr).mean(axis=2) * 3.0, 0, 1)  # |err| row, 3x gain
        cols.append(np.concatenate([gt, pr, np.stack([er, er, er], axis=2)], axis=0))
    grid = (np.clip(np.concatenate(cols, axis=1), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(grid).save(sdir / f"{scene}_vis_GTtop_PREDbot.png")
    _gd = os.environ.get("BENCH_GRID_DIR")
    if _gd:
        os.makedirs(_gd, exist_ok=True)
        _t = os.environ.get("BENCH_GRID_TAG", "")
        Image.fromarray(grid).save(os.path.join(_gd, f"{_t}__{scene}.png" if _t else f"{scene}.png"))

    n_rows = len(rows)
    del enc, ctx, preds_small, rows, data, cond, targets, tgt_items
    if dataset == "kubric4d":
        k4._META_CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return scene_overall, scene_groups, n_rows, peak


def run_scene(recon, scorer, dataset, scene, cfg, out_dir):
    if NEO_FIX:
        H, W = FIX_HW
        prep = center_crop_to_tensor
    elif NEO_FIX_RES:
        H, W = res_match_hw(cfg["native_hw"])
        prep = resize_to_tensor  # native aspect, no crop
    else:
        H, W = C.render_hw_override(cfg.get("native_hw") or cfg["render_hw"], cfg["render_hw"], 14)
        prep = resize_to_tensor

    if NEO_REF:
        return _run_scene_ref(recon, scorer, dataset, scene, cfg, out_dir, H, W, prep)

    groups = cfg["groups"]
    data = load_scene(dataset, scene)
    cond, targets = data["cond"], data["targets"]
    log_t = logical_time_fn(cond)

    ctx = torch.stack([prep(c["image"], H, W) for c in cond])
    ctx_ts = torch.arange(len(cond), dtype=torch.int64)

    tgt_items, group_of = [], []
    for g in groups:
        for t in targets.get(g, []):
            tgt_items.append(t)
            group_of.append(g)
    tgt = torch.stack([prep(t["gt"], H, W) for t in tgt_items])
    tgt_ts = torch.tensor([log_t(t["frame"]) for t in tgt_items], dtype=torch.int64)

    # ---- two-pass encode (predicts target cameras from RGB) ----
    for attempt in range(6):
        try:
            wait_for_gpu(cfg["min_free"])
            torch.cuda.reset_peak_memory_stats()
            encoded = recon.encode_for_evaluation(
                context_images=ctx, context_timestamps=ctx_ts,
                target_images=tgt, target_timestamps=tgt_ts, use_motion=True)
            break
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            print(f"  [OOM encode attempt {attempt}] retry in 30s", flush=True)
            time.sleep(30)
    else:
        raise RuntimeError(f"{scene}: encode OOM after retries")

    sdir = out_dir / scene
    (sdir / "pred").mkdir(parents=True, exist_ok=True)

    rows = []
    preds_small = {}
    n = len(tgt_items)
    vis_idx = np.linspace(0, n - 1, min(8, n)).round().astype(int).tolist()
    chunk = 8
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        for attempt in range(6):
            try:
                pred = recon.render(
                    encoded,
                    c2w_target=encoded["target_c2w"][s:e],
                    k_target=encoded["target_intrinsics"][s:e],
                    timestamps_target=encoded["target_timestamps"][s:e],
                    height=H, width=W)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [OOM render {s}:{e} attempt {attempt}] retry in 30s", flush=True)
                time.sleep(30)
        else:
            raise RuntimeError(f"{scene}: render OOM {s}:{e}")

        for j in range(e - s):
            idx = s + j
            item = tgt_items[idx]
            g = group_of[idx]
            phwc = to_hwc(pred[j])
            covis = item.get("covis_mask", None)
            if NEO_FIX:
                gt = center_crop_np(item["gt"], H, W)
                dyn = center_crop_mask(item.get("dyn_mask", None), H, W) if cfg["has_dyn"] else None
                if covis is not None:
                    covis = center_crop_mask(covis, H, W)
            else:
                gt = np.clip(np.asarray(item["gt"], np.float32), 0, 1)
                dyn = item.get("dyn_mask", None) if cfg["has_dyn"] else None
            m = scorer.score(phwc, gt, dyn_mask=dyn, covis_mask=covis)
            row = {"group": g, "idx": idx, "frame": item.get("frame"),
                   "psnr": m["psnr"], "ssim": m["ssim"], "lpips": m["lpips"]}
            if "offset" in item:
                row["offset"] = item["offset"]
            if "dyn_psnr" in m:
                row["dyn_psnr"] = m["dyn_psnr"]
            rows.append(row)
            # save pred at native GT resolution
            _tag = f"v{C._vtag(item)}_f{int(item['frame']):02d}"
            _gh, _gw = gt.shape[:2]
            _out = np.clip(phwc, 0, 1)
            if _out.shape[:2] != (_gh, _gw):
                _out = np.clip(cv2.resize(_out, (_gw, _gh), interpolation=cv2.INTER_LINEAR), 0, 1)
            Image.fromarray((_out * 255 + 0.5).astype(np.uint8)).save(
                sdir / "pred" / f"{_tag}.png")
            if idx in vis_idx:
                preds_small[idx] = phwc
        del pred
        torch.cuda.empty_cache()

    metric_keys = ["psnr", "ssim", "lpips"] + (["dyn_psnr"] if cfg["has_dyn"] else [])
    scene_groups = {}
    for g in groups:
        grows = [r for r in rows if r["group"] == g]
        if grows:
            scene_groups[g] = agg(grows, metric_keys)
    scene_overall = agg(rows, metric_keys)

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    metrics = {"scene": scene, "dataset": dataset, "render_hw": [H, W],
               "n": len(rows), "peak_gib": peak,
               "overall": scene_overall, **scene_groups, "rows": rows}
    json.dump(metrics, open(sdir / "metrics.json", "w"), indent=2)

    # ---- vis grid: GT / pred / |err| ----
    cols = []
    for i in vis_idx:
        gt = np.clip(np.asarray(tgt_items[i]["gt"], np.float32), 0, 1)   # native GT, as scored
        _gh, _gw = gt.shape[:2]
        pr = np.clip(cv2.resize(np.clip(preds_small[i], 0, 1).astype(np.float32), (_gw, _gh),
                                interpolation=cv2.INTER_LINEAR), 0, 1)
        er = np.clip(np.abs(gt - pr).mean(axis=2) * 3.0, 0, 1)  # |err| row, 3x gain
        cols.append(np.concatenate([gt, pr, np.stack([er, er, er], axis=2)], axis=0))
    grid = (np.clip(np.concatenate(cols, axis=1), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(grid).save(sdir / f"{scene}_vis_GTtop_PREDbot.png")
    _gd = os.environ.get("BENCH_GRID_DIR")
    if _gd:
        os.makedirs(_gd, exist_ok=True)
        _t = os.environ.get("BENCH_GRID_TAG", "")
        Image.fromarray(grid).save(os.path.join(_gd, f"{_t}__{scene}.png" if _t else f"{scene}.png"))

    n_rows = len(rows)
    del encoded, ctx, tgt, preds_small, rows, data, cond, targets, tgt_items
    if dataset == "kubric4d":
        k4._META_CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return scene_overall, scene_groups, n_rows, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--max_scenes", type=int, default=None)
    args = ap.parse_args()

    # NEO_STATIC defaults to static reconstruction except for novel-time datasets.
    if "NEO_STATIC" not in os.environ:
        os.environ["NEO_STATIC"] = "0" if args.dataset in ("adt", "dycheck", "tum") else "1"

    cfg = DATASETS[args.dataset]
    scenes = args.scenes if args.scenes else scene_list(args.dataset)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    out_dir = OUT_BASE / args.dataset / os.environ.get("NEO_OUT_NAME", "neoverse")
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W = C.render_hw_override(cfg.get("native_hw") or cfg["render_hw"], cfg["render_hw"], 14)
    nh, nw = cfg["native_hw"]
    print(f"=== NeoVerse {args.dataset} | render {H}x{W} -> score native {nh}x{nw} ===")
    print(f"GPU visible: {torch.cuda.get_device_name(0)}  free {gpu_free_mib():.0f} MiB", flush=True)

    recon = NeoVerseReconstructor(str(CKPT), device="cuda", dtype=torch.bfloat16)
    scorer = Scorer(device="cuda")

    metric_keys = ["psnr", "ssim", "lpips"] + (["dyn_psnr"] if cfg["has_dyn"] else [])
    per_scene = {}
    scene_overalls = []
    group_scene_means = {g: [] for g in cfg["groups"]}

    for i, sc in enumerate(scenes, 1):
        mpath = out_dir / sc / "metrics.json"
        if mpath.is_file():
            try:
                md = json.load(open(mpath))
                if md.get("overall") and md.get("n", 0) > 0:
                    so = md["overall"]
                    sg = {g: md[g] for g in cfg["groups"] if g in md}
                    per_scene[sc] = {**sg, "overall": so}
                    scene_overalls.append(so)
                    for g in cfg["groups"]:
                        if g in sg:
                            group_scene_means[g].append(sg[g])
                    print(f"[{i}/{len(scenes)}] {sc:12s} RESUMED (overall PSNR {so['psnr']:.3f})",
                          flush=True)
                    continue
            except Exception:
                pass
        try:
            so, sg, n, peak = run_scene(recon, scorer, args.dataset, sc, cfg, out_dir)
            per_scene[sc] = {**sg, "overall": so}
            scene_overalls.append(so)
            for g in cfg["groups"]:
                if g in sg:
                    group_scene_means[g].append(sg[g])
            dyn = f"  dyn {so['dyn_psnr']:.3f}" if so.get("dyn_psnr") is not None else ""
            gstr = "  ".join(
                f"{g}:{sg[g]['psnr']:.3f}" for g in cfg["groups"] if g in sg)
            print(f"[{i}/{len(scenes)}] {sc:12s} overall PSNR {so['psnr']:.3f} "
                  f"SSIM {so['ssim']:.4f} LPIPS {so['lpips']:.4f}{dyn}  [{gstr}]  "
                  f"(n={n}, peak {peak:.1f} GiB)", flush=True)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"{sc}: FAILED {ex}", flush=True)

    def mean_of(dicts):
        return {k: float(np.mean([d[k] for d in dicts
                                  if d.get(k) is not None and np.isfinite(d[k])]))
                if any(d.get(k) is not None and np.isfinite(d[k]) for d in dicts) else None
                for k in metric_keys}

    overall = mean_of(scene_overalls) if scene_overalls else {}
    groups_summary = {g: mean_of(v) for g, v in group_scene_means.items() if v}

    framing = {
        "render_hw": [H, W],
        "gt_hw": [nh, nw],
        "input": f"full-frame {H}x{W} native-aspect (no crop, /14), pose-free",
        "score_res": f"native {nh}x{nw}, pred bilinear-upsampled to GT",
        "lpips": "vgg",
        "pose": (f"reference: context-only Gaussians + GT target poses "
                 f"(align={NEO_ALIGN}, norm_pose={NEO_NORM_POSE}, "
                 f"norm_scale={NEO_NORM_SCALE}, tto={NEO_TTO})"
                 if NEO_REF else "pose-free (NeoVerse two-pass predicted cameras)"),
        "time_token": "logical cond-time index 0..7 (cond + target)",
        "dyn_psnr": bool(cfg["has_dyn"]),
    }
    summary = {
        "model": "NeoVerse",
        "method": "reconstructor render only (no diffusion)",
        "dataset": args.dataset,
        "protocol": "eval_protocol.md unified feed-forward NVS",
        "overall": overall,
        **groups_summary,
        "per_scene": per_scene,
        "framing": framing,
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)

    print(f"\n===== {args.dataset} SUMMARY (n_scenes={len(per_scene)}) =====")
    print("OVERALL", {k: (round(v, 4) if v is not None else None) for k, v in overall.items()})
    for g, gm in groups_summary.items():
        print(f"  {g:12s}", {k: (round(v, 4) if v is not None else None) for k, v in gm.items()})
    print("saved ->", out_dir / "summary.json", flush=True)


if __name__ == "__main__":
    main()
