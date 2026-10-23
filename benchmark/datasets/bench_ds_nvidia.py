"""NVIDIA Dynamic Scenes loader."""

import os

import cv2
import numpy as np
from PIL import Image

from bench_paths import dataset  # noqa: E402
NVIDIA_ROOT = str(dataset("NVIDIA"))

NVIDIA_SCENES = [
    "Balloon1", "Balloon2", "DynamicFace", "Jumping", "Playground",
    "Skating", "Teadybear", "Truck", "Umbrella",
]

NATIVE_HW = (540, 960)          # (H, W)
N_CAMS = 12
CLIP = 12                       # number of multiview_GT time dirs
COND_CAMS = [0, 1, 3, 4]        # rig {0,1,3,4} -> 2
TGT_CAM = 2
K = 4


def _read_calib_nvidia(scene_root, cam):
    """cam index (0-based) -> (K 3x3 native px, c2w 4x4 OpenCV)."""
    cdir = os.path.join(scene_root, "calibration", f"cam{cam + 1:02d}")
    Kmat = np.loadtxt(os.path.join(cdir, "intrinsic.txt"), dtype=np.float32)
    ext = np.loadtxt(os.path.join(cdir, "extrinsic.txt"), dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = ext[1:].T
    c2w[:3, 3] = ext[0]
    return Kmat, c2w


def _gt_root(scene_root):
    root = os.path.join(scene_root, "multiview_GT")
    if not os.path.isdir(os.path.join(root, "00000001")):
        root = os.path.join(root, "multiview_GT")
    return root


def _load_grid_img(gt_root, cam, t):
    """camera `cam` @ time `t` (both 0-based) -> RGB float32 [0,1], HxWx3 native."""
    path = os.path.join(gt_root, f"{t + 1:08d}", f"cam{cam + 1:02d}.jpg")
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def _times(clip=CLIP, k=K):
    """round(linspace(0, clip-1, k)) -> uniform-time cond diagonal indices."""
    return [int(round(x)) for x in np.linspace(0, clip - 1, k)]


def load_nvidia_nvs(scene: str) -> dict:
    """Load one NVIDIA scene for Table 2 (novel view, held-out camera)."""
    scene_root = os.path.join(NVIDIA_ROOT, scene)
    gt_root = _gt_root(scene_root)

    # NVS-8: held-out interior camera, conditioned on its 8 nearest cameras
    import bench_nvs8 as V8
    if V8.NVS8:
        centers = {c: _read_calib_nvidia(scene_root, c)[1][:3, 3] for c in range(N_CAMS)}
        cond_cams, tgt_cam = V8.window9(list(range(N_CAMS)), 1)
        times = V8.times8(CLIP)
        k = len(cond_cams)
    else:
        cond_cams, tgt_cam, times, k = list(COND_CAMS), TGT_CAM, _times(), K

    if os.environ.get("T2_COND_IDX"):
        _ix = [int(x) for x in os.environ["T2_COND_IDX"].split(",")]
        cond_cams = [cond_cams[i] for i in _ix]
    if os.environ.get("T2_COND_TIMES"):
        times = [int(x) for x in os.environ["T2_COND_TIMES"].split(",")]
    k = len(cond_cams)
    assert len(times) == k, "T2 cond times must match cond cam count"
    tgt_times = ([int(x) for x in os.environ["T2_TGT_TIMES"].split(",")]
                 if os.environ.get("T2_TGT_TIMES") else list(times))

    calib = {c: _read_calib_nvidia(scene_root, c) for c in set(cond_cams + [tgt_cam])}

    # cond: view i = cam[cond_cams[i]] @ times[i]
    cond = []
    for i in range(k):
        cam = cond_cams[i]
        t = times[i]
        Kmat, c2w = calib[cam]
        cond.append({
            "image": _load_grid_img(gt_root, cam, t),
            "c2w": c2w.copy(),
            "K": Kmat.copy(),
            "frame": int(t),
            "view": int(cam),
        })

    # targets: held-out camera at the target times (default: cond times)
    Kt, c2wt = calib[tgt_cam]
    targets = []
    for i in range(len(tgt_times)):
        t = tgt_times[i]
        targets.append({
            "c2w": c2wt.copy(),
            "K": Kt.copy(),
            "gt": _load_grid_img(gt_root, tgt_cam, t),
            "dyn_mask": None,
            "frame": int(t),
            "view": int(tgt_cam),
        })

    meta = {
        "native_hw": NATIVE_HW,
        "table": "T2",
        "scene": scene,
        "cond_cams": list(cond_cams),
        "tgt_cam": tgt_cam,
        "times": list(times),
        "clip": CLIP,
        "k": k,
    }
    return {"cond": cond, "targets": {"all": targets}, "meta": meta}


def load_nvidia_ctvs(scene: str) -> dict:
    """NVIDIA CTVS (Table 1): K=12 diagonal cond (cam c @ t=c) + K×K off-diagonal targets = 132."""
    scene_root = os.path.join(NVIDIA_ROOT, scene)
    gt_root = _gt_root(scene_root)
    cams = list(range(12))
    calib = {c: _read_calib_nvidia(scene_root, c) for c in cams}
    cond = []
    for c in cams:                                   # diagonal input: cam c @ time c
        Kmat, c2w = calib[c]
        cond.append({"image": _load_grid_img(gt_root, c, c), "c2w": c2w.copy(),
                     "K": Kmat.copy(), "frame": int(c), "view": int(c)})
    targets = []
    for cam in cams:                                 # K×K off-diagonal (cam @ time), cam != time
        Kmat, c2w = calib[cam]
        for t in cams:
            if cam == t:
                continue
            targets.append({"c2w": c2w.copy(), "K": Kmat.copy(),
                            "gt": _load_grid_img(gt_root, cam, t), "dyn_mask": None,
                            "frame": int(t), "view": int(cam)})
    assert len(targets) == 132, len(targets)
    meta = {"native_hw": NATIVE_HW, "table": "T1", "scene": scene, "k": 12, "clip": CLIP}
    return {"cond": cond, "targets": {"all": targets}, "meta": meta}


NVSVID_K = 8


def nvsvid_indices(n_cams, clip, k=NVSVID_K):
    """(cond_cams, times, tgt_cam): K diagonal cond cams + the held-out middle camera."""
    tgt = n_cams // 2
    pool = [c for c in range(n_cams) if c != tgt]
    idx = np.round(np.linspace(0, len(pool) - 1, k)).astype(int)
    cond_cams = [pool[i] for i in idx]
    times = np.round(np.linspace(0, clip - 1, k)).astype(int).tolist()
    return cond_cams, times, tgt


def load_nvidia_nvsvid(scene: str) -> dict:
    """Novel-view video: K=8 diagonal cond + held-out middle cam at all 8 cond times."""
    scene_root = os.path.join(NVIDIA_ROOT, scene)
    gt_root = _gt_root(scene_root)
    cond_cams, times, tgt_cam = nvsvid_indices(N_CAMS, CLIP)
    assert tgt_cam not in cond_cams
    assert min(cond_cams) < tgt_cam < max(cond_cams), "target must be interior to cond span"
    calib = {c: _read_calib_nvidia(scene_root, c) for c in set(cond_cams + [tgt_cam])}
    cond = []
    for i, cam in enumerate(cond_cams):
        Kmat, c2w = calib[cam]
        cond.append({"image": _load_grid_img(gt_root, cam, times[i]), "c2w": c2w.copy(),
                     "K": Kmat.copy(), "frame": int(times[i]), "view": int(cam)})
    Kt, c2wt = calib[tgt_cam]
    targets = [{"c2w": c2wt.copy(), "K": Kt.copy(),
                "gt": _load_grid_img(gt_root, tgt_cam, t), "dyn_mask": None,
                "frame": int(t), "view": int(tgt_cam)} for t in times]
    meta = {"native_hw": NATIVE_HW, "table": "NVSVID", "scene": scene, "k": NVSVID_K,
            "clip": CLIP, "cond_cams": cond_cams, "cond_times": times, "tgt_cam": tgt_cam}
    return {"cond": cond, "targets": {"all": targets}, "meta": meta}


def load_nvidia(scene: str) -> dict:
    """Dispatch: BENCH_NVSVID -> novel-view video; BENCH_TABLE2 -> T2; else CTVS (132 K×K)."""
    if os.environ.get("BENCH_NVSVID"):
        return load_nvidia_nvsvid(scene)
    return load_nvidia_nvs(scene) if os.environ.get("BENCH_TABLE2") else load_nvidia_ctvs(scene)
