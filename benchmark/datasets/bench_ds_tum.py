"""TUM RGB-D (freiburg dynamic) loader for single-trajectory temporal interpolation."""
import os
import numpy as np
from PIL import Image

from bench_paths import dataset  # noqa: E402
TUM_ROOT = str(dataset("TUM"))
TUM_SCENES = [
    "rgbd_dataset_freiburg3_sitting_static", "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_sitting_halfsphere", "rgbd_dataset_freiburg3_sitting_rpy",
    "rgbd_dataset_freiburg3_walking_static", "rgbd_dataset_freiburg3_walking_xyz",
    "rgbd_dataset_freiburg3_walking_halfsphere", "rgbd_dataset_freiburg3_walking_rpy",
    "rgbd_dataset_freiburg2_desk_with_person",
]
N_CTX = int(os.environ.get("TUM_NCTX", "16"))
# TUM_DT: conditioning stride
DT = int(os.environ.get("TUM_DT", "6"))
START = 0
N_FRAMES = int(os.environ.get("TUM_NFRAMES", str(15 * DT + 1)))

_FR3 = dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6)
_FR2 = dict(fx=520.9, fy=521.0, cx=325.1, cy=249.7)


def _K(scene):
    i = _FR2 if "freiburg2" in scene else _FR3
    return np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], np.float32)


def _quat_to_R(q):
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float64)


def _slerp(q0, q1, t):
    q0 = np.asarray(q0, float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(d); th = th0 * t
    q2 = q1 - q0 * d; q2 /= np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


def _load_gt(seq_dir):
    gt = []
    for line in open(os.path.join(seq_dir, "groundtruth.txt")):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 8:
            gt.append([float(x) for x in p[:8]])       # ts tx ty tz qx qy qz qw
    return np.array(gt)


def _pose_at(gt, ts):
    tss = gt[:, 0]
    j = int(np.clip(np.searchsorted(tss, ts), 1, len(gt) - 1))
    t0, t1 = tss[j - 1], tss[j]
    a = 0.0 if t1 == t0 else float(np.clip((ts - t0) / (t1 - t0), 0, 1))
    tr = gt[j - 1, 1:4] * (1 - a) + gt[j, 1:4] * a
    q = _slerp(gt[j - 1, 4:8], gt[j, 4:8], a)           # (x,y,z,w)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _quat_to_R(q)
    c2w[:3, 3] = tr
    return c2w


def _assoc(seq_dir):
    out = []
    for line in open(os.path.join(seq_dir, "associations.txt")):
        p = line.split()
        if len(p) >= 2:
            out.append((float(p[0]), p[1]))            # (rgb_ts, rgb_path)
    return out


def load_tum_scene(scene, root=TUM_ROOT, dt=DT, start=START):
    sd = os.path.join(root, scene)
    assoc = _assoc(sd)[:N_FRAMES]
    gt = _load_gt(sd)
    K = _K(scene)
    ctx_f = [start + i * dt for i in range(N_CTX)]
    if ctx_f[-1] >= len(assoc):
        raise RuntimeError(f"{scene}: need pos {ctx_f[-1]} but only {len(assoc)} assoc frames")
    if os.environ.get("TUM_FULL_GRID") == "1":
        tgt_f = [f for i in range(N_CTX - 1) for f in range(ctx_f[i] + 1, ctx_f[i + 1])]
    elif os.environ.get("NTS3"):
        _offs = sorted({round(k * dt / 4) for k in (1, 2, 3)})
        tgt_f = [start + i * dt + o for i in range(N_CTX - 1) for o in _offs]
    else:
        tgt_f = [start + i * dt + dt // 2 for i in range(N_CTX - 1)]

    def _img(pos):
        return np.asarray(Image.open(os.path.join(sd, assoc[pos][1])).convert("RGB"), np.float32) / 255.0

    cond = []
    for fr in ctx_f:
        cond.append({"image": _img(fr), "c2w": _pose_at(gt, assoc[fr][0]), "K": K.copy(),
                     "view": 0, "frame": int(fr), "time": (fr - start) / dt})
    targets = {"all": []}
    for fr in tgt_f:
        off = min(fr - ctx_f[(fr - start) // dt], ctx_f[(fr - start) // dt + 1] - fr)
        targets["all"].append({"gt": _img(fr), "c2w": _pose_at(gt, assoc[fr][0]), "K": K.copy(),
                               "view": 0, "frame": int(fr), "time": (fr - start) / dt,
                               "offset": int(off), "dyn_mask": None})
    meta = {"scene": scene, "dt": dt, "n_cond": len(cond), "n_tgt": len(targets["all"]),
            "native_hw": (480, 640)}
    return {"cond": cond, "targets": targets, "meta": meta}
