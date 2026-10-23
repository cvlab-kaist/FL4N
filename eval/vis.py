"""Visualization grids logged during training/validation."""
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _tensor_to_uint8_chw(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float in [0,1] → (H, W, 3) uint8."""
    t = t.detach().to(torch.float32).clamp(0, 1).cpu().numpy()
    t = (t * 255.0).round().astype(np.uint8)
    return np.transpose(t, (1, 2, 0))


def _hcat(imgs: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate (H,W,3) uint8 images horizontally with a 4-pixel white gutter."""
    if not imgs:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    H = max(im.shape[0] for im in imgs)
    gutter = np.full((H, 4, 3), 255, dtype=np.uint8)
    canvas = []
    for i, im in enumerate(imgs):
        if im.shape[0] != H:
            # white pad at the bottom
            pad = np.full((H - im.shape[0], im.shape[1], 3), 255, dtype=np.uint8)
            im = np.concatenate([im, pad], axis=0)
        if i > 0:
            canvas.append(gutter)
        canvas.append(im)
    return np.concatenate(canvas, axis=1)


def _vcat(imgs: Sequence[np.ndarray]) -> np.ndarray:
    if not imgs:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    W = max(im.shape[1] for im in imgs)
    gutter = np.full((4, W, 3), 255, dtype=np.uint8)
    canvas = []
    for i, im in enumerate(imgs):
        if im.shape[1] != W:
            pad = np.full((im.shape[0], W - im.shape[1], 3), 255, dtype=np.uint8)
            im = np.concatenate([im, pad], axis=1)
        if i > 0:
            canvas.append(gutter)
        canvas.append(im)
    return np.concatenate(canvas, axis=0)


def _parse_view_frame(image_id: str):
    """Try to extract (view, frame) from a `..._<view>_<frame>` style image_id. Returns (None, None) if unparseable."""
    parts = image_id.rsplit("_", 2)
    if len(parts) < 3:
        return None, None
    try:
        return int(parts[-2]), int(parts[-1])
    except ValueError:
        return None, None


def _label_for(image_id: Optional[str], tau_val: Optional[float]) -> str:
    v, _ = _parse_view_frame(image_id) if image_id is not None else (None, None)
    t = int(round(tau_val)) if tau_val is not None else None
    if v is not None and t is not None:
        return f"v={v} t={t}"
    if t is not None:
        return f"t={t}"
    if v is not None:
        return f"v={v}"
    return ""


def _render_text_strip(labels: Sequence[str], cell_w: int, gutter: int, height: int = 24) -> np.ndarray:
    """White strip (H, n*cell_w + (n-1)*gutter, 3) with one label per cell aligned to the hcat layout."""
    n = len(labels)
    if n == 0:
        return np.full((height, 1, 3), 255, dtype=np.uint8)
    total_w = n * cell_w + (n - 1) * gutter
    strip = Image.new("RGB", (total_w, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        x = i * (cell_w + gutter) + 6
        draw.text((x, 4), label, fill=(0, 0, 0), font=font)
    return np.array(strip)


def make_compare_grid(
    images: torch.Tensor,      # (V, 3, H, W) gt: first K = cond, rest = target_gt
    output: torch.Tensor,      # (V, 3, H, W) pred: target_pred at [K:]
    num_cond_views: int,
    tau: Optional[torch.Tensor] = None,        # (V,) window-relative offsets, optional for label
    image_ids: Optional[Sequence[str]] = None, # per-view image id (parsed for view), optional for label
    dyn_mask: Optional[torch.Tensor] = None,   # (V, 1, H, W) dynamic-object mask; adds a cond overlay row
    pred_mask: Optional[torch.Tensor] = None,  # (V, 1, H, W) predicted dyn mask; adds a green overlay row
    static_pred: Optional[torch.Tensor] = None,  # (V, 3, H, W) render from the static feature alone; adds a last row
) -> np.ndarray:
    """Row strip: cond / [cond GT dyn-mask] / [cond pred dyn-mask] / target_gt / target_pred / [static_pred]."""
    K = num_cond_views
    cond = images[:K]
    tgt_gt = images[K:]
    tgt_pred = output[K:]
    n_tgt = tgt_gt.shape[0]

    with_label = tau is not None or image_ids is not None
    cell_w = cond.shape[3]

    def _maybe_label(img: np.ndarray, view_idx: int) -> np.ndarray:
        if not with_label:
            return img
        label = _label_for(
            image_ids[view_idx] if image_ids is not None else None,
            float(tau[view_idx]) if tau is not None else None,
        )
        return _vcat([_render_text_strip([label], cell_w, 0), img])

    def _cell(frame: torch.Tensor, view_idx: int) -> np.ndarray:
        return _maybe_label(_tensor_to_uint8_chw(frame), view_idx)

    def _cell_mask(frame: torch.Tensor, mask: torch.Tensor, view_idx: int, color=(255.0, 0.0, 0.0)) -> np.ndarray:
        img = _tensor_to_uint8_chw(frame).astype(np.float32)
        m = (mask[0].detach().to(torch.float32).cpu().numpy() > 0.5)   # (H, W) bool
        img[m] = 0.35 * img[m] + 0.65 * np.array(color, dtype=np.float32)  # tint on dynamic px
        return _maybe_label(img.clip(0, 255).astype(np.uint8), view_idx)

    rows = [_hcat([_cell(cond[i], i) for i in range(K)])]
    if dyn_mask is not None:
        rows.append(_hcat([_cell_mask(cond[i], dyn_mask[i], i) for i in range(K)]))
    if pred_mask is not None:   # predicted dyn mask (green) on cond views
        rows.append(_hcat([_cell_mask(cond[i], pred_mask[i], i, color=(0.0, 220.0, 0.0)) for i in range(K)]))
    rows.append(_hcat([_cell(tgt_gt[i], K + i) for i in range(n_tgt)]))
    rows.append(_hcat([_cell(tgt_pred[i], K + i) for i in range(n_tgt)]))
    if static_pred is not None:   # last row: target render from the static feature alone
        rows.append(_hcat([_cell(static_pred[K + i], K + i) for i in range(n_tgt)]))

    return _vcat(rows)


def _overlay_c(rgb, mask, color):
    img = _tensor_to_uint8_chw(rgb).astype(np.float32)
    m = mask.detach().to(torch.float32).cpu().numpy(); m = (m[0] if m.ndim == 3 else m) > 0.5
    for c in range(3):
        img[..., c] = img[..., c] * (1.0 - 0.55 * m) + color[c] * 0.55 * m
    return img.astype(np.uint8)


def make_grid_vis(images, output, num_cond_views, n_self, n_others, n_novel,
                  static_pred=None, gt_mask=None, pred_mask=None, title=""):
    """[GT | PRED | STATIC] N x N grids over views [cond K][self K][others K*(K-1)][novel ...], plus novel and dyn-mask strips; untargeted cells are black."""
    K = int(num_cond_views)
    H, W = images.shape[-2], images.shape[-1]
    black = np.zeros((H, W, 3), np.uint8)
    cond = images[:K]
    off = [(i, j) for i in range(K) for j in range(K) if i != j][:n_others]
    obase = K + n_self                                   # others start after cond + self
    gt_at, pr_at, st_at = {}, {}, {}
    for i in range(K):
        gt_at[(i, i)] = _tensor_to_uint8_chw(cond[i]); pr_at[(i, i)] = _tensor_to_uint8_chw(cond[i])
        st_at[(i, i)] = _tensor_to_uint8_chw(static_pred[i]) if static_pred is not None else black
    for m, (i, j) in enumerate(off):
        gt_at[(i, j)] = _tensor_to_uint8_chw(images[obase + m])
        pr_at[(i, j)] = _tensor_to_uint8_chw(output[obase + m])
        if static_pred is not None:
            st_at[(i, j)] = _tensor_to_uint8_chw(static_pred[obase + m])

    def bd(img, kind, b=3):
        col = {"input": (30, 175, 60), "target": (205, 65, 65), "empty": (70, 70, 70)}[kind]
        c = np.empty((img.shape[0] + 2 * b, img.shape[1] + 2 * b, 3), np.uint8); c[:] = np.array(col, np.uint8)
        c[b:b + img.shape[0], b:b + img.shape[1]] = img; return c

    def grid(at, diag_in):
        rows = []
        for i in range(K):
            cells = []
            for j in range(K):
                if (i, j) in at:
                    cells.append(bd(at[(i, j)], "input" if (i == j and diag_in) else "target"))
                else:
                    cells.append(bd(black, "empty"))
            rows.append(_hcat(cells))
        return _vcat(rows)

    gt_b = _vcat([_render_text_strip([f"GROUND TRUTH (diag=input) {title}"], 0, 0, 22), grid(gt_at, True)])
    pr_b = _vcat([_render_text_strip([f"PREDICTION (diag=input) {title}"], 0, 0, 22), grid(pr_at, True)])
    blocks = [gt_b, pr_b]
    if static_pred is not None:
        blocks.append(_vcat([_render_text_strip([f"STATIC-ONLY {title}"], 0, 0, 22), grid(st_at, False)]))
    sep = lambda h: np.full((h, 14, 3), 255, np.uint8)
    hs = max(b.shape[0] for b in blocks)
    top = _hcat(sum([[b, sep(hs)] for b in blocks], [])[:-1])

    out_rows = [top]
    nbase = obase + n_others
    if n_novel > 0:
        ng = _hcat([bd(_tensor_to_uint8_chw(images[nbase + i]), "target") for i in range(n_novel)])
        npr = _hcat([bd(_tensor_to_uint8_chw(output[nbase + i]), "target") for i in range(n_novel)])
        rows_n = [_render_text_strip(["NOVEL-VIEW/TIME GT"], 0, 0, 22), ng,
                  _render_text_strip(["NOVEL-VIEW/TIME PRED"], 0, 0, 22), npr]
        if static_pred is not None:
            ns = _hcat([bd(_tensor_to_uint8_chw(static_pred[nbase + i]), "target") for i in range(n_novel)])
            rows_n += [_render_text_strip(["NOVEL STATIC-ONLY"], 0, 0, 22), ns]
        out_rows.append(_vcat(rows_n))

    if gt_mask is not None:
        out_rows.append(_vcat([_render_text_strip(["DYN MASK GT (per cond cam)"], 0, 0, 22),
                               _hcat([bd(_overlay_c(cond[i], gt_mask[i], (230, 40, 40)), "input") for i in range(K)])]))
    if pred_mask is not None:
        out_rows.append(_vcat([_render_text_strip(["DYN MASK PRED (per cond cam)"], 0, 0, 22),
                               _hcat([bd(_overlay_c(cond[i], pred_mask[i], (30, 220, 30)), "input") for i in range(K)])]))
    return _vcat(out_rows)


def _shrink(arr: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return arr
    h, w = arr.shape[:2]
    return np.asarray(Image.fromarray(arr).resize((max(1, int(w * scale)), max(1, int(h * scale)))))


def make_nvidia_grid(
    cond: torch.Tensor,            # (K, 3, H, W) monocular input frames (diagonal cam@time)
    tgt_gt: torch.Tensor,          # (C, T, 3, H, W) GT per target cam over T timesteps
    tgt_pred: torch.Tensor,        # (C, T, 3, H, W) prediction
    tgt_static: Optional[torch.Tensor],  # (C, T, 3, H, W) static-only render or None
    gt_mask: Optional[torch.Tensor],     # (K, 1, H, W) GT dyn mask on input frames
    pred_mask: Optional[torch.Tensor],   # (K, 1, H, W) predicted dyn mask on input frames
    target_cam_labels: Sequence[str],    # e.g. ["cam01","cam07","cam12"]
    time_labels: Sequence[int],
    title: str = "",
    cell_scale: float = 0.5,
) -> np.ndarray:
    """Rows (each a T-wide strip): input / dyn mask GT / dyn mask PRED / then per target cam (GT / PRED / STATIC)."""
    K = cond.shape[0]
    H, W = int(cond.shape[-2] * cell_scale), int(cond.shape[-1] * cell_scale)
    gray = np.full((H, W, 3), 60, np.uint8)
    def img(t): return _shrink(_tensor_to_uint8_chw(t), cell_scale)
    def ov(rgb, m, c): return _shrink(_overlay_c(rgb, m, c), cell_scale)
    rows, labels = [], []
    rows.append([img(cond[i]) for i in range(K)]); labels.append("input")
    rows.append([ov(cond[i], gt_mask[i], (230, 40, 40)) for i in range(K)] if gt_mask is not None else [gray] * K); labels.append("dyn mask GT")
    rows.append([ov(cond[i], pred_mask[i], (30, 220, 30)) for i in range(K)] if pred_mask is not None else [gray] * K); labels.append("dyn mask PRED")
    C, T = tgt_gt.shape[0], tgt_gt.shape[1]
    for ci in range(C):
        rows.append([img(tgt_gt[ci, t]) for t in range(T)]); labels.append(f"{target_cam_labels[ci]} GT")
        rows.append([img(tgt_pred[ci, t]) for t in range(T)]); labels.append(f"{target_cam_labels[ci]} PRED")
        rows.append([img(tgt_static[ci, t]) for t in range(T)] if tgt_static is not None else [gray] * T); labels.append(f"{target_cam_labels[ci]} STATIC")
    body = _vcat([_hcat(r) for r in rows])
    header = _render_text_strip([f"t{time_labels[j]}" for j in range(len(rows[0]))], W, 4, height=22)
    stacked = _vcat([header, body])
    pitch = H + 4
    lab_col = Image.new("RGB", (150, stacked.shape[0]), (255, 255, 255))
    draw = ImageDraw.Draw(lab_col)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    if title:
        draw.text((4, 4), title, fill=(0, 0, 0), font=font)
    for r, lab in enumerate(labels):
        draw.text((4, 22 + r * pitch + H // 2 - 7), lab, fill=(0, 0, 0), font=font)
    return _hcat([np.array(lab_col), stacked])
