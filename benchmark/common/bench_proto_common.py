#!/usr/bin/env python3
"""Shared helpers for the protocol NVS evals."""
import json
import os

import cv2
import numpy as np
from PIL import Image

METRIC_KEYS = ("psnr", "ssim", "lpips", "dyn_psnr")


# Resolution / intrinsics
def render_hw_override(gt_hw, default_hw, patch):
    """BENCH_LONG_EDGE=N -> aspect-preserving fit with longest side N, rounded to `patch`."""
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
