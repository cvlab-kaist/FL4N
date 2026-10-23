"""CamXTime dataset loader."""

import os
from bench_static_util import maybe_static  # noqa: E402
import json

import numpy as np
import bench_nvs8 as _V8

try:
    import cv2
except Exception as _e:  # pragma: no cover
    cv2 = None

from bench_paths import dataset  # noqa: E402
ROOT = str(dataset("CamxTime"))
NATIVE = 512                       # native square resolution
N_FRAMES = 120                     # time frames per mp4
K_COND = int(os.environ.get("CX_K", "8"))          # conditioning views
_SPAN = int(os.environ.get("CX_SPAN", "56"))       # cond extent in slots (stride*(K-1))
_auto = max(1, _SPAN // max(1, K_COND - 1)) if K_COND > 1 else 8
CAM_STRIDE = int(os.environ.get("CX_STRIDE", str(_auto)))    # camera-position stride
FRAME_STRIDE = int(os.environ.get("CX_STRIDE", str(_auto)))  # temporal stride
TRAIN_FRAC = 0.9                   # 10% test holdout

CAM_KEY_OFFSET = 32                # trajectory JSON key = str(32 + cam_number)
_BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

COND_TRAJ = "camera-trajectory-01"

T2_TRAJ = COND_TRAJ
K_COND_T2 = 8


# Pose math
def _blender_c2w_to_opencv(c2w):
    """Blender world c2w -> OpenCV c2w (negate rotation columns 1 and 2)."""
    out = np.array(c2w, dtype=np.float32).copy()
    out[:3, :3] = out[:3, :3] @ _BLENDER_TO_OPENCV
    return out


# Cross-scene test split
def _usable_scenes(root):
    """Scenes with a trajectory that has its camera JSON and >=2 cam*_full_motion.mp4 files."""
    scenes = []
    for scene in sorted(os.listdir(root)):
        sdir = os.path.join(root, scene)
        if not (scene.lower().startswith("scene") and os.path.isdir(sdir)):
            continue
        ok = False
        for traj in sorted(os.listdir(sdir)):
            tdir = os.path.join(sdir, traj)
            if not (traj.startswith("camera-trajectory-") and os.path.isdir(tdir)):
                continue
            jpath = os.path.join(tdir, f"{traj}-camera.json")
            if not os.path.isfile(jpath):
                continue
            n_mp4 = sum(
                1 for f in os.listdir(tdir)
                if f.startswith("cam") and f.endswith("_full_motion.mp4")
            )
            if n_mp4 >= 2:
                ok = True
                break
        if ok:
            scenes.append(scene)
    return scenes


def list_test_scenes(root=ROOT):
    """Cross-scene 10% holdout: sorted scenes, cut=int(N*0.9), test=scenes[cut:]."""
    scenes = _usable_scenes(root)
    cut = int(len(scenes) * TRAIN_FRAC)
    return scenes[cut:]


# Trajectory / camera helpers
def _traj_dir(root, scene, traj):
    return os.path.join(root, scene, traj)


def _read_traj_json(root, scene, traj):
    p = os.path.join(_traj_dir(root, scene, traj), f"{traj}-camera.json")
    with open(p) as f:
        return json.load(f)


def _cam_numbers(root, scene, traj):
    """Sorted camera numbers (1-indexed) present as mp4 in the trajectory folder."""
    tdir = _traj_dir(root, scene, traj)
    nums = []
    for f in sorted(os.listdir(tdir)):
        if f.startswith("cam") and f.endswith("_full_motion.mp4"):
            nums.append(int(f[3:6]))
    return nums


def _mp4_path(root, scene, traj, cam_number):
    return os.path.join(_traj_dir(root, scene, traj), f"cam{cam_number:03d}_full_motion.mp4")


def _K_from_json(js):
    """3x3 intrinsics at native 512x512."""
    return np.array(js["intrinsics"]["K"], dtype=np.float32)


def _pose_opencv(js, cam_number):
    """OpenCV c2w (4x4) for a camera position; pose is time-independent."""
    traj = js["trajectory"]
    key = str(CAM_KEY_OFFSET + cam_number)
    if key not in traj:
        traj_keys = sorted(traj, key=int)
        key = traj_keys[(cam_number - 1) % len(traj_keys)]
    return _blender_c2w_to_opencv(traj[key]["c2w"])


# Protocol index selection
def _cond_indices(num_cam=N_FRAMES, num_frame=N_FRAMES, k=K_COND):
    """Evenly strided camera positions and frames; cond view i = cam[i] @ t[i]."""
    cam_slots = [CAM_STRIDE * i for i in range(k)]
    frames = [FRAME_STRIDE * i for i in range(k)]
    if cam_slots[-1] >= num_cam or frames[-1] >= num_frame:
        raise RuntimeError(f"camxtime: stride {CAM_STRIDE}/{FRAME_STRIDE} x k={k} exceeds {num_cam}/{num_frame}")
    return cam_slots, frames


def _transport_plan(cam_slots, frames):
    """CTVS targets: every off-diagonal (cond camera i, cond time j != i) cell."""
    k = len(cam_slots)
    return [(cam_slots[i], frames[j]) for i in range(k) for j in range(k) if i != j]


# Frame decoding
def _decode_frames(mp4_path, frame_idxs):
    """Return {frame_idx: HxWx3 float32 RGB [0,1]} for the requested indices."""
    if cv2 is None:
        raise RuntimeError("cv2 unavailable; cannot decode mp4")
    want = sorted({int(t) for t in frame_idxs})
    max_t = want[-1]
    want_set = set(want)
    out = {}
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    i = 0
    try:
        while i <= max_t:
            ok, bgr = cap.read()
            if not ok:
                break
            if i in want_set:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                out[i] = (rgb.astype(np.float32) / 255.0)
            i += 1
    finally:
        cap.release()
    if not out:
        raise RuntimeError(f"decoded no frames from {mp4_path}")
    last = out[max(out)]
    return {t: out.get(t, last) for t in want}   # fall back to the last frame for short clips


def load_camxtime_scene(scene, root=ROOT):
    """Load one CamXTime scene into protocol tensors."""
    # Table 2: novel view (held-out interior trajectory)
    if os.environ.get("BENCH_TABLE2"):
        js = _read_traj_json(root, scene, T2_TRAJ)
        cams = _cam_numbers(root, scene, T2_TRAJ)
        Kmat = _K_from_json(js)
        cond_slots, tgt_slot = _V8.window9(list(range(len(cams))), CAM_STRIDE)
        times = [FRAME_STRIDE * i for i in range(K_COND_T2)]
        if os.environ.get("T2_COND_IDX"):
            _ix = [int(x) for x in os.environ["T2_COND_IDX"].split(",")]
            cond_slots = [cond_slots[i] for i in _ix]
        if os.environ.get("T2_COND_TIMES"):
            times = [int(x) for x in os.environ["T2_COND_TIMES"].split(",")]
        assert len(times) == len(cond_slots), "T2 cond times must match cond slot count"
        tgt_times = ([int(x) for x in os.environ["T2_TGT_TIMES"].split(",")]
                     if os.environ.get("T2_TGT_TIMES") else list(times))
        need = {}
        for i, sl in enumerate(cond_slots):
            need.setdefault(cams[sl], set()).add(times[i])
        need.setdefault(cams[tgt_slot], set()).update(tgt_times)
        dec = {cn: _decode_frames(_mp4_path(root, scene, T2_TRAJ, cn), fs) for cn, fs in need.items()}
        cond = []
        for i, sl in enumerate(cond_slots):
            cn = cams[sl]
            cond.append({"image": dec[cn][times[i]], "c2w": _pose_opencv(js, cn), "K": Kmat.copy(),
                         "cam": sl, "frame": times[i], "traj": T2_TRAJ, "cam_number": cn})
        cn = cams[tgt_slot]
        targets_all = [{"gt": dec[cn][t], "c2w": _pose_opencv(js, cn), "K": Kmat.copy(),
                        "cam": tgt_slot, "frame": t, "dyn_mask": None,
                        "traj": T2_TRAJ, "cam_number": cn} for t in tgt_times]
        meta = {"scene": scene, "native_hw": (NATIVE, NATIVE), "table": "T2", "traj": T2_TRAJ,
                "cond_slots": cond_slots, "tgt_slot": tgt_slot, "times": times, "dyn_mask": None}
        return maybe_static({"cond": cond, "targets": {"all": targets_all}, "meta": meta})

    js = _read_traj_json(root, scene, COND_TRAJ)
    cams = _cam_numbers(root, scene, COND_TRAJ)
    cam_slots, frames = _cond_indices(num_cam=len(cams), num_frame=N_FRAMES, k=K_COND)
    plan = _transport_plan(cam_slots, frames)
    Kmat = _K_from_json(js)

    need = {}
    for slot, fr in list(zip(cam_slots, frames)) + plan:
        need.setdefault(cams[slot], set()).add(int(fr))
    dec = {cn: _decode_frames(_mp4_path(root, scene, COND_TRAJ, cn), fs) for cn, fs in need.items()}

    def _item(slot, fr):
        cn = cams[slot]
        return {"cam": int(slot), "frame": int(fr), "c2w": _pose_opencv(js, cn), "K": Kmat.copy(),
                "traj": COND_TRAJ, "cam_number": int(cn)}

    cond = [{**_item(sl, fr), "image": dec[cams[sl]][fr]} for sl, fr in zip(cam_slots, frames)]
    targets = [{**_item(sl, fr), "gt": dec[cams[sl]][fr], "dyn_mask": None} for sl, fr in plan]
    meta = {"scene": scene, "root": root, "native_hw": (NATIVE, NATIVE), "num_frames": N_FRAMES,
            "traj": COND_TRAJ, "cond_cam_slots": list(cam_slots), "cond_frames": list(frames),
            "known_time": True, "dyn_mask": None, "split": {"mode": "cross_scene", "train_frac": TRAIN_FRAC}}
    return maybe_static({"cond": cond, "targets": {"all": targets}, "meta": meta})


# Validation: load one test scene, print stats, save a montage
def _cam_center(c2w):
    return np.asarray(c2w)[:3, 3]
