"""Charge dataset loader."""

import json
from bench_static_util import maybe_static  # noqa: E402
import os
from typing import List, Optional

import numpy as np
from PIL import Image

from bench_paths import dataset  # noqa: E402
ROOT = str(dataset("Charge_v1_0"))
SCENES = [
    "010_0050", "020_0020", "040_0040", "050_0130",
    "050_0160", "060_0100", "060_0130", "070_0123",
]
NATIVE_HW = (858, 2048)          # (H, W)
NUM_COND = 8
CAM_STRIDE = 1
# Canonical order of the 10 static test cams (coverage priority, not a path).
TRAJ_ORDER = [5, 1, 4, 6, 2, 8, 0, 7, 3, 9]
FRAME_STRIDE = 8
NUM_TEST_CAMS = 10


def _blender_c2w_to_opencv(c2w) -> np.ndarray:
    """Blender/OpenGL c2w -> OpenCV c2w by flipping the Y and Z basis columns."""
    out = np.asarray(c2w, dtype=np.float64).copy()
    out[:3, 1:3] *= -1.0
    return out


def _load_rgb(path: str, resize_hw: Optional[tuple]) -> np.ndarray:
    """RGB image as float32 [0,1], HxWx3. Optionally bilinear-resized to (H,W)."""
    im = Image.open(path).convert("RGB")
    if resize_hw is not None and (im.size[1], im.size[0]) != tuple(resize_hw):
        im = im.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR)
    return (np.asarray(im, dtype=np.float32) / 255.0)


def _load_dyn_mask(path: str, hw: tuple) -> np.ndarray:
    """Boolean HxW dyn mask (True = dynamic). Missing file -> all-False."""
    if os.path.isfile(path):
        m = Image.open(path).convert("L")
        if (m.size[1], m.size[0]) != tuple(hw):
            m = m.resize((hw[1], hw[0]), Image.NEAREST)
        return np.asarray(m) > 127
    return np.zeros(hw, dtype=bool)


def _mask_path(img_path: str) -> str:
    return img_path[:-4] + "_dyn_mask.png" if img_path.endswith(".png") else img_path + "_dyn_mask.png"


def _cond_time_indices(clip: int, k: int = NUM_COND) -> List[int]:
    """k timesteps at fixed stride FRAME_STRIDE from frame 0."""
    idx = [FRAME_STRIDE * i for i in range(k)]
    if idx[-1] >= clip:
        raise RuntimeError(f"charge: stride {FRAME_STRIDE} x k={k} needs {idx[-1] + 1} frames, clip has {clip}")
    return idx


def load_charge_scene(scene: str, resize_hw: Optional[tuple] = None) -> dict:
    """Load one Charge scene per protocol."""
    sparse = os.path.join(ROOT, scene, "Sparse")

    # Table 2: novel view (held-out interior camera)
    if os.environ.get("BENCH_TABLE2"):
        te_path = os.path.join(sparse, "transforms_test.json")
        if not os.path.isfile(te_path):
            raise FileNotFoundError(f"[Charge] {scene}: missing transforms_test.json")
        test = json.load(open(te_path))
        test_cams = sorted(
            c for c in test
            if c.startswith("Sparse_Test") and os.path.isdir(os.path.join(sparse, c))
        )
        if len(test_cams) < 5:
            raise RuntimeError(
                f"[Charge] {scene}: need >=5 Sparse_Test cams for Table 2, got {len(test_cams)}"
            )
        clip = len(test[test_cams[0]])
        win = [TRAJ_ORDER[CAM_STRIDE * i] for i in range(9)]
        cond_cam_idx = [c for j, c in enumerate(win) if j != 4]
        target_cam_idx = win[4]
        times = _cond_time_indices(clip, NUM_COND)

        cond = []
        for i, ci in enumerate(cond_cam_idx):
            cam = test_cams[ci]
            e = test[cam][times[i]]
            img_path = os.path.join(sparse, e["image_path"])
            cond.append({
                "cam": ci,
                "cam_name": cam,
                "frame": int(e["frame_id"]),
                "image": _load_rgb(img_path, resize_hw),
                "c2w": _blender_c2w_to_opencv(e["transformation_matrix"]),
                "K": np.asarray(e["K"], dtype=np.float64),
            })

        tgt_cam = test_cams[target_cam_idx]
        targets_all = []
        for ti in times:
            e = test[tgt_cam][ti]
            img_path = os.path.join(sparse, e["image_path"])
            targets_all.append({
                "cam": target_cam_idx,
                "cam_name": tgt_cam,
                "frame": int(e["frame_id"]),
                "c2w": _blender_c2w_to_opencv(e["transformation_matrix"]),
                "K": np.asarray(e["K"], dtype=np.float64),
                "gt": _load_rgb(img_path, None),
                "dyn_mask": _load_dyn_mask(_mask_path(img_path), NATIVE_HW),
            })

        meta = {
            "scene": scene,
            "native_hw": NATIVE_HW,
            "table": "T2",
            "layout": "{0,1,3,4}->2",
            "n_frames": clip,
            "cond_cam_idx": cond_cam_idx,
            "target_cam_idx": target_cam_idx,
            "cond_times": times,
            "test_cams": test_cams,
        }
        return maybe_static({"cond": cond, "targets": {"all": targets_all}, "meta": meta})

    tr_path = os.path.join(sparse, "transforms_train.json")
    if not os.path.isfile(tr_path):
        tr_path = os.path.join(sparse, "transforms_train_9views.json")
    te_path = os.path.join(sparse, "transforms_test.json")
    if not os.path.isfile(te_path):
        raise FileNotFoundError(f"[Charge] {scene}: missing transforms_test.json")
    train = json.load(open(tr_path))
    test = json.load(open(te_path))

    # cond camera: first Sparse_Train camera present on disk
    train_cams = sorted(
        c for c in train
        if c.startswith("Sparse_Train") and os.path.isdir(os.path.join(sparse, c))
    )
    if not train_cams:
        raise RuntimeError(f"[Charge] {scene}: no Sparse_Train camera dirs on disk")
    cond_cam = train_cams[0]
    cond_entries = train[cond_cam]
    clip = len(cond_entries)                       # synced frame count for this scene
    cond_times = _cond_time_indices(clip, NUM_COND)

    # CTVS: 8 strided test cams on an 8-time diagonal; targets are the 56 off-diagonal cells
    test_cams = sorted(
        c for c in test
        if c.startswith("Sparse_Test") and os.path.isdir(os.path.join(sparse, c))
    )
    if len(test_cams) < NUM_COND:
        raise RuntimeError(f"[Charge] {scene}: need >={NUM_COND} Sparse_Test cams, got {len(test_cams)}")
    clip = len(test[test_cams[0]])
    cond_times = _cond_time_indices(clip, NUM_COND)
    cam_slots = [TRAJ_ORDER[CAM_STRIDE * i] for i in range(NUM_COND)]
    diag_cams = [test_cams[i] for i in cam_slots]

    cond = []
    for i, cam in enumerate(diag_cams):
        e = test[cam][cond_times[i]]
        cond.append({
            "cam": cam,
            "frame": int(e["frame_id"]),
            "image": _load_rgb(os.path.join(sparse, e["image_path"]), resize_hw),
            "c2w": _blender_c2w_to_opencv(e["transformation_matrix"]),
            "K": np.asarray(e["K"], dtype=np.float64),
        })

    targets_all = []
    for i, cam in enumerate(diag_cams):
        for j, tj in enumerate(cond_times):
            if i == j:
                continue
            e = test[cam][tj]
            ip = os.path.join(sparse, e["image_path"])
            targets_all.append({
                "cam": cam,
                "frame": int(e["frame_id"]),
                "c2w": _blender_c2w_to_opencv(e["transformation_matrix"]),
                "K": np.asarray(e["K"], dtype=np.float64),
                "gt": _load_rgb(ip, None),
                "dyn_mask": _load_dyn_mask(_mask_path(ip), NATIVE_HW),
            })

    meta = {
        "scene": scene,
        "native_hw": NATIVE_HW,
        "n_test_cams": len(test_cams),
        "n_frames": clip,
        "cond_cams": diag_cams,
        "cond_times": cond_times,
        "test_cams": test_cams,
        "table": "CTVS",
    }
    return maybe_static({"cond": cond, "targets": {"all": targets_all}, "meta": meta})


# ----------------------------------------------------------------------------- validation
