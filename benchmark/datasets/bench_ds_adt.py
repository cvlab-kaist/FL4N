"""ADT (Aria Digital Twin) loader for single-trajectory temporal interpolation."""
import os, json
import numpy as np
from PIL import Image

from bench_paths import dataset  # noqa: E402
ADT_ROOT = str(dataset("ADT"))
ADT_SCENES = [
    "Apartment_release_meal_skeleton_seq135_M1292",
    "Apartment_release_multiskeleton_party_seq114_M1292",
    "Apartment_release_multiuser_cook_seq141_M1292",
    "Apartment_release_work_skeleton_seq137_M1292",
]
N_CTX = 16
DT = 6
START = 0
RGB_STREAM = "camera-rgb"

_prov = {}


def _provider(vrs):
    if vrs not in _prov:
        import projectaria_tools.core.data_provider as dp
        _prov[vrs] = dp.create_vrs_data_provider(vrs)
    return _prov[vrs]


def _read_rgb(seq_dir, frame_idx, intr):
    """RGB frame -> undistorted linear pinhole [H,W,3] float[0,1]."""
    cache = os.path.join(seq_dir, "frames_rgb", f"{int(frame_idx):06d}.png")
    if os.path.exists(cache):
        return np.asarray(Image.open(cache).convert("RGB"), np.float32) / 255.0
    from projectaria_tools.core import calibration as cal
    prov = _provider(os.path.join(seq_dir, "video.vrs"))
    sid = prov.get_stream_id_from_label(RGB_STREAM)
    src = prov.get_device_calibration().get_camera_calib(RGB_STREAM)
    img = prov.get_image_data_by_index(sid, int(frame_idx))[0].to_numpy_array()
    w, h, f = intr["rgb"]["w"], intr["rgb"]["h"], intr["rgb_focal"]
    lin = cal.get_linear_camera_calibration(w, h, f, RGB_STREAM, src.get_transform_device_camera())
    u = cal.distort_by_calibration(img, lin, src)
    if u.ndim == 2:
        u = np.stack([u] * 3, -1)
    u = np.clip(u.astype(np.float32), 0, 255)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    Image.fromarray(u.astype(np.uint8)).save(cache)
    return u / 255.0


def load_adt_scene(scene, root=ADT_ROOT, dt=DT, start=START):
    sd = os.path.join(root, scene)
    pre = os.path.join(sd, "preprocessed")
    poses = np.load(os.path.join(pre, "poses_rgb.npy")).astype(np.float32)   # [N,4,4] c2w
    intr = json.load(open(os.path.join(pre, "intrinsics.json")))
    K = np.array([[intr["rgb"]["fx"], 0, intr["rgb"]["cx"]],
                  [0, intr["rgb"]["fy"], intr["rgb"]["cy"]], [0, 0, 1]], np.float32)
    N = len(poses)
    ctx_f = [start + i * dt for i in range(N_CTX)]
    if ctx_f[-1] >= N:
        raise RuntimeError(f"{scene}: need frame {ctx_f[-1]} but only {N} poses")
    if os.environ.get("ADT_FULL_GRID") == "1":
        tgt_f = [f for i in range(N_CTX - 1) for f in range(ctx_f[i] + 1, ctx_f[i + 1])]
    else:
        tgt_f = [start + i * dt + dt // 2 for i in range(N_CTX - 1)]

    cond = []
    for fr in ctx_f:
        cond.append({"image": _read_rgb(sd, fr, intr), "c2w": poses[fr].copy(), "K": K.copy(),
                     "view": 0, "frame": int(fr), "time": (fr - start) / dt})
    targets = {"all": []}
    for fr in tgt_f:
        off = min(fr - ctx_f[(fr - start) // dt], ctx_f[(fr - start) // dt + 1] - fr)   # dist to nearest cond
        targets["all"].append({"gt": _read_rgb(sd, fr, intr), "c2w": poses[fr].copy(), "K": K.copy(),
                               "view": 0, "frame": int(fr), "time": (fr - start) / dt,
                               "offset": int(off), "dyn_mask": None})
    meta = {"scene": scene, "dt": dt, "n_cond": len(cond), "n_tgt": len(targets["all"]),
            "native_hw": (intr["rgb"]["h"], intr["rgb"]["w"])}
    return {"cond": cond, "targets": targets, "meta": meta}
