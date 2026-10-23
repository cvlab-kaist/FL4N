"""Kubric4D dataset loader."""

import os
from bench_static_util import maybe_static  # noqa: E402
import json

import numpy as np
from PIL import Image


from bench_paths import dataset  # noqa: E402
KUBRIC4D_ROOT = str(dataset("Kubric4D") / "test")

# 30 curated target scenes (fixed order).
KUBRIC4D_SCENES = [
    "scn02952", "scn02977", "scn02917", "scn02959", "scn02990", "scn02998",
    "scn02911", "scn02945", "scn02935", "scn02966", "scn02921", "scn02915",
    "scn02941", "scn02946", "scn02956", "scn02932", "scn02910", "scn02925",
    "scn02983", "scn02939", "scn02902", "scn02926", "scn02981", "scn02904",
    "scn02993", "scn02991", "scn02940", "scn02914", "scn02928", "scn02978",
]

_ALL_ON_DISK = sorted(d for d in os.listdir(KUBRIC4D_ROOT) if d.startswith("scn"))
if os.environ.get("KUB4D_SCENES"):
    KUBRIC4D_SCENES = [x.strip() for x in os.environ["KUB4D_SCENES"].split(",") if x.strip()]
elif os.environ.get("KUB4D_ALL") == "1":
    KUBRIC4D_SCENES = _ALL_ON_DISK

N_VIEWS = 16
N_FRAMES = 60
FRAME_STRIDE = 8
NATIVE_W, NATIVE_H = 576, 384

# Protocol (view, frame) index sets
COND_SPECS = [(4 + i, 8 * i) for i in range(8)]
_COND_CAMS = [c for c, _ in COND_SPECS]
_COND_TIMES = [t for _, t in COND_SPECS]
TARGET_ALL = [(_COND_CAMS[i], _COND_TIMES[j])
              for i in range(len(COND_SPECS)) for j in range(len(COND_SPECS)) if i != j]

# dyn-mask motion thresholds
VEL_THRESH = 0.03        # per-clip-frame translation (world units) -> "moving"
ROT_STEP_THRESH = 6.0    # per-clip-frame rotation (deg) -> "moving"

_BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


# Geometry
def _quat_wxyz_to_R(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _blender_c2w_to_opencv(pos, quat_wxyz):
    """Kubric per-frame (position, wxyz quaternion, Blender) -> 4x4 OpenCV c2w."""
    R_blender = _quat_wxyz_to_R(quat_wxyz)
    R_cv = R_blender @ _BLENDER_TO_OPENCV
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_cv
    c2w[:3, 3] = np.asarray(pos, dtype=np.float64)
    return c2w


def _K_from_json(K_norm, im_w=NATIVE_W, im_h=NATIVE_H):
    """Normalized/Blender-signed JSON K -> 3x3 pixel intrinsics at (im_w, im_h)."""
    fx = float(K_norm[0][0]) * im_w
    fy = abs(float(K_norm[1][1])) * im_h
    cx = -float(K_norm[0][2]) * im_w
    cy = -float(K_norm[1][2]) * im_h
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _pair_angle_deg(q0, q1):
    """Geodesic angle (deg) between two wxyz quaternions."""
    q0 = np.asarray(q0, np.float64); q1 = np.asarray(q1, np.float64)
    q0 = q0 / (np.linalg.norm(q0) + 1e-8)
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(float(q0 @ q1)), 0.0, 1.0))))


# I/O helpers
def _view_dir(scene_dir, view):
    return os.path.join(scene_dir, f"frames_p0_v{view}")


def _json_path(scene_dir, scene, view):
    return os.path.join(scene_dir, f"{scene}_p0_v{view}.json")


_META_CACHE = {}


def _read_meta(path):
    md = _META_CACHE.get(path)
    if md is None:
        with open(path) as f:
            md = json.load(f)
        _META_CACHE[path] = md
    return md


def _load_rgb(scene_dir, view, frame):
    """rgba_NNNNN.png -> HxWx3 float32 in [0,1] (drops alpha)."""
    p = os.path.join(_view_dir(scene_dir, view), f"rgba_{frame:05d}.png")
    im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return im


def _moving_ids(md, frame):
    """seg ids (instance idx + 1) of instances MOVING at `frame` (motion vs a neighbor)."""
    t_prev = frame - 1 if frame > 0 else min(frame + 1, N_FRAMES - 1)
    gap = max(1, abs(frame - t_prev))
    ids = set()
    for i, o in enumerate(md.get("instances", [])):
        p = np.asarray(o.get("positions", []), dtype=np.float64)
        if p.shape[0] <= max(frame, t_prev):
            continue
        trans = float(np.linalg.norm(p[frame] - p[t_prev])) / gap
        q = np.asarray(o.get("quaternions", []), dtype=np.float64)
        rot = _pair_angle_deg(q[frame], q[t_prev]) / gap if q.shape[0] > max(frame, t_prev) else 0.0
        if trans > VEL_THRESH or rot > ROT_STEP_THRESH:
            ids.add(i + 1)          # seg value = instance index + 1 (0 = background)
    return ids


def _dyn_mask(scene_dir, scene, view, frame):
    """Boolean HxW dyn mask = GT segmentation intersect moving-instance ids, for one target."""
    md = _read_meta(_json_path(scene_dir, scene, view))
    ids = _moving_ids(md, frame)
    seg_p = os.path.join(_view_dir(scene_dir, view), f"segmentation_{frame:05d}.png")
    seg = np.asarray(Image.open(seg_p))
    if seg.ndim == 3:
        seg = seg[..., 0]
    if not ids:
        return np.zeros(seg.shape, dtype=bool)
    return np.isin(seg, list(ids))


def _cam(scene_dir, scene, view, frame):
    """(c2w 4x4 OpenCV, K 3x3) for one (view, frame)."""
    cam = _read_meta(_json_path(scene_dir, scene, view))["camera"]
    c2w = _blender_c2w_to_opencv(cam["positions"][frame], cam["quaternions"][frame])
    K = _K_from_json(cam["K"])
    return c2w, K


# Public loader
def load_kubric4d_scene(scene: str, root: str = KUBRIC4D_ROOT) -> dict:
    """Load one Kubric4D scene."""
    scene_dir = os.path.join(root, scene)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Kubric4D scene dir missing: {scene_dir}")

    # Table 2: novel view (held-out camera)
    if os.environ.get("BENCH_TABLE2"):
        import bench_nvs8 as V8
        if V8.NVS8:
            RING = list(range(4, N_VIEWS))          # horizontal ring v4..v15 (v0-v3 are top-down)
            t2_cond_cams, t2_tgt_cam = V8.window9(RING, 1, start=4)
            t2_times = [FRAME_STRIDE * i for i in range(8)]
        else:
            t2_cond_cams = [4, 5, 7, 8]
            t2_tgt_cam = 6
            t2_times = [int(round(x)) for x in np.linspace(0, N_FRAMES - 1, 4)]
        if os.environ.get("T2_COND_IDX"):
            _ix = [int(x) for x in os.environ["T2_COND_IDX"].split(",")]
            t2_cond_cams = [t2_cond_cams[i] for i in _ix]
        if os.environ.get("T2_COND_TIMES"):
            t2_times = [int(x) for x in os.environ["T2_COND_TIMES"].split(",")]
        assert len(t2_times) == len(t2_cond_cams), "T2 cond times must match cond cam count"
        t2_tgt_times = ([int(x) for x in os.environ["T2_TGT_TIMES"].split(",")]
                        if os.environ.get("T2_TGT_TIMES") else list(t2_times))
        t2_cond = []
        for c, t in zip(t2_cond_cams, t2_times):
            c2w, K = _cam(scene_dir, scene, c, t)
            t2_cond.append({
                "image": _load_rgb(scene_dir, c, t),
                "c2w": c2w,
                "K": K,
                "view": c,
                "frame": t,
            })
        t2_targets = {"all": []}
        for t in t2_tgt_times:
            c2w, K = _cam(scene_dir, scene, t2_tgt_cam, t)
            t2_targets["all"].append({
                "view": t2_tgt_cam,
                "frame": t,
                "c2w": c2w,
                "K": K,
                "gt": _load_rgb(scene_dir, t2_tgt_cam, t),
                "dyn_mask": _dyn_mask(scene_dir, scene, t2_tgt_cam, t),
            })
        t2_meta = {
            "n_views": N_VIEWS,
            "n_frames": N_FRAMES,
            "native_hw": (NATIVE_H, NATIVE_W),
            "scene": scene,
            "root": root,
            "table": "T2",
        }
        return maybe_static({"cond": t2_cond, "targets": t2_targets, "meta": t2_meta})

    # cond: posed monocular diagonal
    cond = []
    for v, t in COND_SPECS:
        c2w, K = _cam(scene_dir, scene, v, t)
        cond.append({
            "image": _load_rgb(scene_dir, v, t),
            "c2w": c2w,
            "K": K,
            "view": v,
            "frame": t,
        })

    def _mk_target(v, t):
        c2w, K = _cam(scene_dir, scene, v, t)
        return {
            "view": v,
            "frame": t,
            "c2w": c2w,
            "K": K,
            "gt": _load_rgb(scene_dir, v, t),
            "dyn_mask": _dyn_mask(scene_dir, scene, v, t),
        }

    targets = {"all": [_mk_target(v, t) for v, t in TARGET_ALL]}

    meta = {
        "n_views": N_VIEWS,
        "n_frames": N_FRAMES,
        "native_hw": (NATIVE_H, NATIVE_W),
        "scene": scene,
        "root": root,
    }
    return maybe_static({"cond": cond, "targets": targets, "meta": meta})
