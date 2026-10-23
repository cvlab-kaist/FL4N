#!/usr/bin/env python3
"""N3DV (DyNeRF) loader for the protocol runners, with a frame cache."""
import glob
import os

import numpy as np

from bench_paths import cache, dataset  # noqa: E402
from bench_static_util import maybe_static  # noqa: E402

N3DV_ROOT = str(dataset("N3DV"))
N3DV_SCENES = [
    "coffee_martini", "cook_spinach", "cut_roasted_beef",
    "flame_steak", "sear_steak", "flame_salmon_1",
]
N3DV_K_COND = 12


# Pose / intrinsics
def llff_to_opencv_c2w(pose_3x5: np.ndarray) -> np.ndarray:
    """LLFF `poses_bounds` 3x5 [R|t|hwf] -> 4x4 OpenCV camera-to-world."""
    R = pose_3x5[:, :3]
    t = pose_3x5[:, 3]
    c0, c1, c2 = R[:, 0], R[:, 1], R[:, 2]
    R_cv = np.stack([c1, c0, -c2], axis=1)  # columns -> [right, down, forward]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_cv
    c2w[:3, 3] = t
    return c2w


def _n3dv_intrinsics(hwf: np.ndarray, eval_hw) -> np.ndarray:
    """3x3 K for eval_hw from LLFF hwf=[H_native,W_native,focal_native]."""
    H_n, W_n, f_n = float(hwf[0]), float(hwf[1]), float(hwf[2])
    H_e, W_e = eval_hw
    sx, sy = W_e / W_n, H_e / H_n
    return np.array([[f_n * sx, 0.0, 0.5 * W_n * sx],
                     [0.0, f_n * sy, 0.5 * H_n * sy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# Protocol index selection
def n3dv_cond_indices(n_cams, clip_len):
    """(cond_cams, cond_times) for the N x N CTVS grid."""
    if os.environ.get("N3DV_ALLCAM", "1") == "0":
        cams = np.round(np.linspace(0, n_cams - 1, N3DV_K_COND)).astype(int).tolist()
        times = np.round(np.linspace(0, clip_len - 1, N3DV_K_COND)).astype(int).tolist()
        return cams, times
    cams = list(range(1, n_cams))                       # all cameras except cam0
    N = len(cams)
    times = np.round(np.linspace(0, clip_len - 1, N)).astype(int).tolist()
    return cams, times


def n3dv_target_indices(n_cams, cond_cams, cond_times):
    """{"novel_view":[(cam,time),...], "transport":[(cam,time),...]}."""
    cond_cam_set = set(cond_cams)
    held_out = [c for c in range(n_cams) if c not in cond_cam_set]
    cond_time_set = sorted(set(cond_times))
    novel = [(c, t) for c in held_out for t in cond_time_set]
    transport = []
    for i, cam in enumerate(cond_cams):
        own_time = cond_times[i]
        for t in cond_times:
            if t != own_time:
                transport.append((cam, t))
    return {"novel_view": novel, "transport": transport}

EVAL_HW = (1014, 1352)          # 2x downscale of native 2028x2704
CACHE_DIR = str(cache("n3dv_frames"))


# Geometry (no frame decode)
def _geometry(scene, eval_hw, root=N3DV_ROOT):
    import cv2
    scene_dir = os.path.join(root, scene)
    cam_files = sorted(glob.glob(os.path.join(scene_dir, "cam*.mp4")))
    if not cam_files:
        raise FileNotFoundError(f"no cam*.mp4 in {scene_dir}")
    n_cams = len(cam_files)
    pb = np.load(os.path.join(scene_dir, "poses_bounds.npy"))
    poses = pb[:, :15].reshape(-1, 3, 5)
    c2ws = [llff_to_opencv_c2w(poses[i]) for i in range(n_cams)]
    Ks = [_n3dv_intrinsics(poses[i, :, 4], eval_hw) for i in range(n_cams)]
    cap = cv2.VideoCapture(cam_files[0])
    clip_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W_n = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_n = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    cond_cams, cond_times = n3dv_cond_indices(n_cams, clip_len)
    # N3DV_CAM_ORDER: optional per-scene spatial camera order (file order is not spatial)
    _ord_p = os.environ.get("N3DV_CAM_ORDER")
    if _ord_p:
        import json as _json
        _o = _json.load(open(_ord_p)).get(scene)
        if _o:
            _seq = _o["continuous_order_fileSlot"]
            _seq = [c for c in _seq if 0 <= c < n_cams]
            _pick = np.round(np.linspace(0, len(_seq) - 1, len(cond_cams))).astype(int)
            cond_cams = [_seq[i] for i in _pick]
            print(f"[n3dv] {scene}: continuous cond cams {cond_cams}", flush=True)
    tgt_idx = n3dv_target_indices(n_cams, cond_cams, cond_times)
    meta = {"scene": scene, "n_cams": n_cams, "clip_len": clip_len,
            "video_native_hw": (H_n, W_n), "eval_hw": tuple(eval_hw),
            "native_hw": tuple(eval_hw), "n_frames": clip_len,
            "num_frames": clip_len, "cond_cams": cond_cams,
            "cond_times": cond_times}
    return cam_files, c2ws, Ks, cond_cams, cond_times, tgt_idx, meta


def _decode_group(cam_files, pairs, eval_hw):
    """Decode (cam, time) pairs -> uint8 (n,H,W,3), opening each camera video once."""
    import cv2
    H_e, W_e = eval_hw
    out = np.empty((len(pairs), H_e, W_e, 3), np.uint8)
    by_cam = {}
    for i, (cam, t) in enumerate(pairs):
        by_cam.setdefault(cam, []).append((t, i))
    for cam, lst in by_cam.items():
        lst.sort()
        cap = cv2.VideoCapture(cam_files[cam])
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {cam_files[cam]}")
        for t, i in lst:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"failed frame {t} of {cam_files[cam]}")
            if (bgr.shape[0], bgr.shape[1]) != (H_e, W_e):
                bgr = cv2.resize(bgr, (W_e, H_e), interpolation=cv2.INTER_AREA)
            out[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out


def _pixels(scene, eval_hw, cam_files, cond_cams, cond_times, tgt_idx):
    os.makedirs(CACHE_DIR, exist_ok=True)
    import hashlib
    sel = repr((sorted(map(int, cond_cams)), sorted(map(int, cond_times)),
                len(tgt_idx.get("transport", [])), len(tgt_idx.get("novel_view", []))))
    h = hashlib.md5(sel.encode()).hexdigest()[:10]
    tag = f"{scene}_{eval_hw[0]}x{eval_hw[1]}_{len(cond_cams)}c_{h}"
    cpath = os.path.join(CACHE_DIR, tag + ".npz")
    if os.path.isfile(cpath):
        z = np.load(cpath)
        return z["cond"], z["novel_view"], z["transport"]
    cond_px = _decode_group(cam_files, list(zip(cond_cams, cond_times)), eval_hw)
    nv_px = _decode_group(cam_files, tgt_idx["novel_view"], eval_hw)
    tr_px = _decode_group(cam_files, tgt_idx["transport"], eval_hw)
    tmp = cpath + ".tmp.npz"
    np.savez(tmp, cond=cond_px, novel_view=nv_px, transport=tr_px)
    os.replace(tmp, cpath)
    return cond_px, nv_px, tr_px


def load_n3dv_scene(scene, eval_hw=EVAL_HW):
    eval_hw = tuple(eval_hw)

    # Novel-view video: K=8 diagonal cond + held-out middle cam at all cond times
    if os.environ.get("BENCH_NVSVID"):
        from bench_ds_nvidia import nvsvid_indices
        cam_files, c2ws, Ks, _cc, _ct, _ti, meta = _geometry(scene, eval_hw)
        cond_cams, times, tgt_cam = nvsvid_indices(meta["n_cams"], meta["clip_len"])
        assert tgt_cam not in cond_cams
        assert min(cond_cams) < tgt_cam < max(cond_cams), "target must be interior to cond span"
        cond_pairs = list(zip(cond_cams, times))
        tgt_pairs = [(tgt_cam, t) for t in times]
        cond_px = _decode_group(cam_files, cond_pairs, eval_hw)
        tgt_px = _decode_group(cam_files, tgt_pairs, eval_hw)
        cond = [{"image": cond_px[i].astype(np.float32) / 255.0,
                 "c2w": np.asarray(c2ws[cam], np.float64), "K": np.asarray(Ks[cam], np.float64),
                 "time": int(t), "frame": int(t), "cam": int(cam), "view": int(cam)}
                for i, (cam, t) in enumerate(cond_pairs)]
        targets_all = [{"cam": int(cam), "view": int(cam), "time": int(t), "frame": int(t),
                        "c2w": np.asarray(c2ws[cam], np.float64),
                        "K": np.asarray(Ks[cam], np.float64),
                        "gt": tgt_px[j].astype(np.float32) / 255.0, "dyn_mask": None}
                       for j, (cam, t) in enumerate(tgt_pairs)]
        nv_meta = dict(meta)
        nv_meta.update({"cond_cams": cond_cams, "cond_times": times,
                        "tgt_cam": tgt_cam, "table": "NVSVID"})
        return maybe_static({"cond": cond, "targets": {"all": targets_all}, "meta": nv_meta})

    # Table 2: novel view (held-out camera)
    if os.environ.get("BENCH_TABLE2"):
        cam_files, c2ws, Ks, _cc, _ct, _ti, meta = _geometry(scene, eval_hw)
        clip_len = meta["clip_len"]
        import bench_nvs8 as V8
        if V8.NVS8:
            centers = {c: np.asarray(c2ws[c])[:3, 3] for c in range(len(c2ws))}
            pool = list(range(len(c2ws)))
            # hold out the interior camera (nearest the rig centroid), condition on its 8 nearest
            if os.environ.get("N3DV_NVS_INTERIOR", "1") == "0":
                t2_cond_cams, t2_tgt_cam = V8.window9(pool, 1)
            else:
                t2_tgt_cam = V8.interior_cam(centers, pool)
                t2_cond_cams = V8.nearest8(centers, t2_tgt_cam, pool)
            t2_times = V8.times8(clip_len)
        else:
            t2_cond_cams = [0, 1, 3, 4]
            t2_tgt_cam = 2
            t2_times = np.round(np.linspace(0, clip_len - 1, 4)).astype(int).tolist()
        cond_pairs = list(zip(t2_cond_cams, t2_times))
        tgt_pairs = [(t2_tgt_cam, t) for t in t2_times]
        cond_px = _decode_group(cam_files, cond_pairs, eval_hw)
        tgt_px = _decode_group(cam_files, tgt_pairs, eval_hw)
        cond = []
        for i, (cam, t) in enumerate(cond_pairs):
            cond.append({
                "image": cond_px[i].astype(np.float32) / 255.0,
                "c2w": np.asarray(c2ws[cam], np.float64),
                "K": np.asarray(Ks[cam], np.float64),
                "time": int(t), "frame": int(t),
                "cam": int(cam), "view": int(cam),
            })
        targets_all = []
        for j, (cam, t) in enumerate(tgt_pairs):
            targets_all.append({
                "cam": int(cam), "view": int(cam),
                "time": int(t), "frame": int(t),
                "c2w": np.asarray(c2ws[cam], np.float64),
                "K": np.asarray(Ks[cam], np.float64),
                "gt": tgt_px[j].astype(np.float32) / 255.0,
                "dyn_mask": None,
            })
        t2_meta = dict(meta)
        t2_meta["cond_cams"] = t2_cond_cams
        t2_meta["cond_times"] = t2_times
        t2_meta["table"] = "T2"
        return maybe_static({"cond": cond, "targets": {"all": targets_all}, "meta": t2_meta})

    cam_files, c2ws, Ks, cond_cams, cond_times, tgt_idx, meta = _geometry(scene, eval_hw)
    cond_px, nv_px, tr_px = _pixels(scene, eval_hw, cam_files,
                                    cond_cams, cond_times, tgt_idx)

    cond = []
    for i in range(len(cond_cams)):
        cam, t = cond_cams[i], cond_times[i]
        cond.append({
            "image": cond_px[i].astype(np.float32) / 255.0,
            "c2w": np.asarray(c2ws[cam], np.float64),
            "K": np.asarray(Ks[cam], np.float64),
            "time": int(t), "frame": int(t),
            "cam": int(cam), "view": int(cam),
        })

    def _mk(pairs, px):
        items = []
        for j, (cam, t) in enumerate(pairs):
            items.append({
                "cam": int(cam), "view": int(cam),
                "time": int(t), "frame": int(t),
                "c2w": np.asarray(c2ws[cam], np.float64),
                "K": np.asarray(Ks[cam], np.float64),
                "gt": px[j].astype(np.float32) / 255.0,
                "dyn_mask": None,
            })
        return items

    targets = {"all": _mk(tgt_idx["transport"], tr_px)}
    return maybe_static({"cond": cond, "targets": targets, "meta": meta})
