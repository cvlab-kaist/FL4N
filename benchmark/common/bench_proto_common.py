#!/usr/bin/env python3
"""Shared helpers for the protocol NVS evals."""
import json
import os

import cv2
import numpy as np
from PIL import Image

METRIC_KEYS = ("psnr", "ssim", "lpips", "dyn_psnr")


# Resolution / intrinsics
def resize_img_K(img, K, H, W):
    """Full-frame resize of img (HxWx3 [0,1]) + rescale intrinsics to (H,W)."""
    sh, sw = img.shape[:2]
    out = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    Ko = K.copy().astype(np.float32)
    Ko[0, 0] *= W / sw; Ko[0, 2] *= W / sw
    Ko[1, 1] *= H / sh; Ko[1, 2] *= H / sh
    return out, Ko


def render_hw_override(gt_hw, default_hw, patch):
    """BENCH_LONG_EDGE=N -> aspect-preserving fit with longest side N, rounded to `patch`."""
    sq = os.environ.get("BENCH_SQUARE")
    if sq:
        n = int(sq)
        return (n, n)
    le = os.environ.get("BENCH_LONG_EDGE")
    if not le:
        return tuple(default_hw)
    n = int(le)
    h, w = gt_hw
    sc = n / float(max(h, w))
    r = lambda v: max(patch, int(round(v * sc / patch)) * patch)
    return (r(h), r(w))


def scale_K(K, sh, sw, H, W):
    """Rescale a native-res (sh,sw) intrinsic to render res (H,W)."""
    Ko = K.copy().astype(np.float32)
    Ko[0, 0] *= W / sw; Ko[0, 2] *= W / sw
    Ko[1, 1] *= H / sh; Ko[1, 2] *= H / sh
    return Ko


def norm_intrinsics(K, H, W):
    """MoVieS fxfycxcy normalized intrinsics."""
    return np.array([K[0, 0] / W, K[1, 1] / H, K[0, 2] / W, K[1, 2] / H], np.float32)


# Pose normalization (fit on cond, apply to any set)
def _scale_mult():
    """BENCH_POSE_SCALE_MULT: multiply the normalised camera scale."""
    try:
        return float(os.environ.get("BENCH_POSE_SCALE_MULT", "1.0"))
    except ValueError:
        return 1.0


def fit_norm_4dgt(cond_c2ws, fallback_c2ws=None):
    """Canonicalize to cam0, then mean-center translations and scale so max |t - mean| == 1."""
    M = np.linalg.inv(cond_c2ws[0].astype(np.float64))
    t = (M[None] @ cond_c2ws.astype(np.float64))[:, :3, 3]
    # BENCH_NORM_REF: upstream 4DGT eval normalization
    if os.environ.get("BENCH_NORM_REF") == "1":
        mean = np.zeros(3)
        r = float(np.linalg.norm(t[1:] - t[0], axis=1).max()) if len(t) > 1 else 0.0
        if r < 1e-6:
            r = 1.0
        return (r, mean), M
    mean = t.mean(axis=0)
    r = float(np.linalg.norm(t - mean, axis=1).max())
    # BENCH_NORM_SPAN_ALL: scale on the cond+target span
    if os.environ.get("BENCH_NORM_SPAN_ALL") == "1" and fallback_c2ws is not None:
        tf = (M[None] @ fallback_c2ws.astype(np.float64))[:, :3, 3]
        r = max(r, float(np.linalg.norm(tf - mean, axis=1).max()))
    if r < 1e-6 and fallback_c2ws is not None:
        tf = (M[None] @ fallback_c2ws.astype(np.float64))[:, :3, 3]
        r = float(np.linalg.norm(tf - mean, axis=1).max())
    if r < 1e-6:
        r = 1.0
    return (r / _scale_mult(), mean), M


def apply_norm_4dgt(c2ws, base, M):
    n = (M[None] @ c2ws.astype(np.float64)).copy()
    r, mean = base if isinstance(base, tuple) else (base, np.zeros(3))
    n[:, :3, 3] = (n[:, :3, 3] - mean) / r
    return n.astype(np.float32)


def fit_norm_movies(cond_c2ws, fallback_c2ws=None):
    """MoVieS: canon cam0 + /median rig radius. Returns (canon 4x4, scale, radius)."""
    centers = cond_c2ws[:, :3, 3]
    # BENCH_NORM_REF: upstream MoVieS baseline_normalize (src/eval.py)
    if os.environ.get("BENCH_NORM_REF") == "1":
        canon = np.linalg.inv(cond_c2ws[0])
        rc = np.einsum("ij,njk->nik", canon, cond_c2ws)[:, :3, 3]
        radius = float(np.linalg.norm(rc[1:] - rc[0], axis=1).max()) if len(rc) > 1 else 0.0
        if radius <= 1e-8:
            radius = 1.0
        return canon, 1.0 / radius, radius
    radius = float(np.median(np.linalg.norm(centers - centers.mean(axis=0), axis=1)))
    if radius <= 1e-8 and fallback_c2ws is not None:
        allc = np.concatenate([centers, fallback_c2ws[:, :3, 3]], axis=0)
        radius = float(np.median(np.linalg.norm(allc - allc.mean(axis=0), axis=1)))
    if radius <= 1e-8:
        radius = 1.0
    return np.linalg.inv(cond_c2ws[0]), _scale_mult() / radius, radius


def apply_norm_movies(c2ws, canon, scale):
    n = np.einsum("ij,njk->nik", canon, c2ws)
    n[:, :3, 3] *= scale
    return n.astype(np.float32)


# Aggregation
def mean_metrics(rows):
    out = {}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in rows if r.get(k) is not None], np.float64)
        vals = vals[~np.isnan(vals)]
        out[k] = float(vals.mean()) if vals.size else float("nan")
    return out


def flatten_items(targets):
    """dict{group:[items]} -> ordered list of (group, item), groups sorted for determinism."""
    order = [g for g in ("transport", "novel_view") if g in targets]
    order += [g for g in targets if g not in order]
    return [(g, it) for g in order for it in targets[g]]


def score_and_save(items, preds, scorer, sdir, scene, render_hw):
    """Score each (group, item) at native GT res; write pred pngs, metrics.json and a vis grid."""
    _nosave = bool(os.environ.get("BENCH_NOSAVE"))
    (sdir / "pred").mkdir(parents=True, exist_ok=True)
    RH, RW = render_hw
    groups = {}
    all_rows = []
    for (group, it), pred in zip(items, preds):
        pred = np.clip(pred, 0, 1)
        m = scorer.score(pred, it["gt"], dyn_mask=it.get("dyn_mask"), covis_mask=it.get("covis_mask"))
        rec = {"group": group, "view": it.get("view", it.get("cam")),
               "frame": it["frame"], **({"offset": it["offset"]} if "offset" in it else {}),
               **{k: m.get(k) for k in METRIC_KEYS}}
        groups.setdefault(group, []).append(rec)
        all_rows.append(rec)
        if not _nosave:
            tag = f"v{_vtag(it)}_f{it['frame']:02d}"
            gh, gw = np.asarray(it["gt"]).shape[:2]
            out = pred if pred.shape[:2] == (gh, gw) else np.clip(
                cv2.resize(pred, (gw, gh), interpolation=cv2.INTER_LINEAR), 0, 1)
            Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(sdir / "pred" / f"{tag}.png")

    agg = {g: mean_metrics(rows) for g, rows in groups.items()}
    agg["overall"] = mean_metrics(all_rows)
    json.dump({"scene": scene, **agg, "rows": all_rows},
              open(sdir / "metrics.json", "w"), indent=2)

    if _nosave:
        return agg
    idx = np.linspace(0, len(items) - 1, min(8, len(items))).round().astype(int)
    cols = []
    for i in idx:
        gt = np.clip(items[i][1]["gt"], 0, 1)
        gh, gw = gt.shape[:2]
        pr = np.clip(cv2.resize(np.clip(preds[i], 0, 1), (gw, gh), interpolation=cv2.INTER_LINEAR), 0, 1)
        # third row: |err| at 3x gain
        er = np.clip(np.abs(gt - pr).mean(axis=2) * 3.0, 0, 1)
        er = np.stack([er, er, er], axis=2)
        cols.append(np.concatenate([gt, pr, er], axis=0))
    grid = (np.clip(np.concatenate(cols, axis=1), 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(grid).save(sdir / f"{scene}_vis_GTtop_PREDbot.png")
    _gd = os.environ.get("BENCH_GRID_DIR")
    if _gd:
        os.makedirs(_gd, exist_ok=True)
        _tag = os.environ.get("BENCH_GRID_TAG", "")
        _nm = f"{_tag}__{scene}.png" if _tag else f"{scene}.png"
        Image.fromarray(grid).save(os.path.join(_gd, _nm))
    return agg


def _vtag(it):
    v = it.get("view", it.get("cam"))
    return f"{v:02d}" if isinstance(v, int) else str(v)


def write_summary(out_dir, model, dataset, per_scene, framing, group_names):
    def grp(g):
        vals = {k: [per_scene[s][g][k] for s in per_scene if g in per_scene[s]]
                for k in METRIC_KEYS}
        return {k: (float(np.nanmean(vals[k])) if vals[k] else float("nan"))
                for k in METRIC_KEYS}
    summary = {"model": model, "dataset": dataset,
               "overall": grp("overall")}
    for g in group_names:
        summary[g] = grp(g)
    summary["per_scene"] = per_scene
    summary["framing"] = framing
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    return summary
