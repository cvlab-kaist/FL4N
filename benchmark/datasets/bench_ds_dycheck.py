"""DyCheck (iPhone) loader."""
import os, json
import numpy as np
from PIL import Image

from bench_paths import dataset  # noqa: E402
DYCHECK_ROOT = str(dataset("DyCheck"))
DYCHECK_SCENES = ["apple", "block", "paper-windmill", "spin", "teddy"]
NUM_CONTEXT = int(os.environ.get("DYCHECK_NUM_CONTEXT", "32"))   # cam0 input frames
MAX_TEMPORAL = int(os.environ.get("DYCHECK_MAX_TEMPORAL", "64")) # window of common timestamps
STRIDE = int(os.environ.get("DYCHECK_STRIDE", "2"))
COND_CAM, TGT_CAM = 0, 1

_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _nerfies_c2w_cv(orientation, position):
    """Nerfies camera JSON -> OpenCV c2w."""
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(orientation, np.float32).T
    c2w[:3, 3] = np.asarray(position, np.float32)
    return c2w


def _common_app_ids(scene_dir):
    """appearance_ids present in every camera (0,1,2), sorted -> the shared timeline."""
    md = json.load(open(os.path.join(scene_dir, "metadata.json")))
    by_cam = {}
    for k, v in md.items():
        by_cam.setdefault(int(v["camera_id"]), set()).add(int(v["appearance_id"]))
    cams = sorted(by_cam)
    common = set(by_cam[cams[0]])
    for c in cams[1:]:
        common &= by_cam[c]
    return sorted(common)


def _cam(scene_dir, cam, app):
    j = json.load(open(os.path.join(scene_dir, "camera", f"{cam}_{app:05d}.json")))
    c2w = _nerfies_c2w_cv(j["orientation"], j["position"])
    f = float(j["focal_length"]); cx, cy = j["principal_point"]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
    return c2w, K


def _rgb(scene_dir, cam, app):
    return np.asarray(Image.open(os.path.join(scene_dir, "rgb", "1x", f"{cam}_{app:05d}.png"))
                      .convert("RGB"), np.float32) / 255.0


def _covis(scene_dir, cam, app):
    p = os.path.join(scene_dir, "covisible", "2x", "val", f"{cam}_{app:05d}.png")
    if not os.path.exists(p):
        return None
    return (np.asarray(Image.open(p).convert("L"), np.float32) > 127)   # HxW bool (2x res)


def load_dycheck_scene(scene, root=DYCHECK_ROOT):
    """Load one DyCheck scene (single-trajectory by default, cross-camera if DYCHECK_CROSSCAM=1)."""
    if os.environ.get("DYCHECK_CROSSCAM") == "1":
        return _load_dycheck_crosscam(scene, root=root)
    return load_dycheck_st_scene(scene, root=root)


def _load_dycheck_crosscam(scene, root=DYCHECK_ROOT):
    sd = os.path.join(root, scene)
    common = _common_app_ids(sd)
    T = len(common)
    max_t = min(T - 1, MAX_TEMPORAL - 1)
    ctx_idx = list(range(0, max_t + 1, STRIDE))[:NUM_CONTEXT]
    last_ctx = ctx_idx[-1]
    tgt_idx = list(range(0, last_ctx + 1))

    # scene normalization (scene.json): p_norm = (p - center) * scale
    sj = json.load(open(os.path.join(sd, "scene.json")))
    center = np.asarray(sj["center"], np.float32); sscale = float(sj["scale"])

    def norm(c2w):
        c2w = c2w.copy(); c2w[:3, 3] = (c2w[:3, 3] - center) * sscale
        return c2w

    cond = []
    for i in ctx_idx:
        a = common[i]; c2w, K = _cam(sd, COND_CAM, a)
        cond.append({"image": _rgb(sd, COND_CAM, a), "c2w": norm(c2w), "K": K,
                     "view": COND_CAM, "frame": int(a), "time": i / 2.0})
    targets = {"all": []}
    for i in tgt_idx:
        a = common[i]; c2w, K = _cam(sd, TGT_CAM, a)
        targets["all"].append({"gt": _rgb(sd, TGT_CAM, a), "c2w": norm(c2w), "K": K,
                               "view": TGT_CAM, "frame": int(a), "time": i / 2.0,
                               "dyn_mask": None, "covis_mask": _covis(sd, TGT_CAM, a)})
    meta = {"scene": scene, "cond_cam": COND_CAM, "tgt_cam": TGT_CAM, "prenormalized": True,
            "n_cond": len(cond), "n_tgt": len(targets["all"]), "native_hw": (960, 720)}
    return {"cond": cond, "targets": targets, "meta": meta}


# Single-trajectory variant: one moving camera (cam0) for cond and targets
ST_N_CTX = int(os.environ.get("ST_N_CTX", "16"))
ST_DT = 4               # NTS temporal stride


def load_dycheck_st_scene(scene, root=DYCHECK_ROOT, dt=ST_DT):
    sd = os.path.join(root, scene)
    common = _common_app_ids(sd)
    T = len(common)
    n_ctx = ST_N_CTX
    while (n_ctx - 1) * dt >= T and n_ctx > 2:      # shrink if the timeline is short
        n_ctx -= 1
    ctx_i = [i * dt for i in range(n_ctx)]

    sj = json.load(open(os.path.join(sd, "scene.json")))
    center = np.asarray(sj["center"], np.float32); sscale = float(sj["scale"])

    def norm(c2w):
        c2w = c2w.copy(); c2w[:3, 3] = (c2w[:3, 3] - center) * sscale
        return c2w

    cond = []
    for i in ctx_i:
        a = common[i]; c2w, K = _cam(sd, COND_CAM, a)
        cond.append({"image": _rgb(sd, COND_CAM, a), "c2w": norm(c2w), "K": K,
                     "view": COND_CAM, "frame": int(a), "time": i / 2.0})
    tgt = []
    _offs = ((1, 2, 3) if os.environ.get("NTS3") else (dt // 2,))
    for k in range(len(ctx_i) - 1):
        for _o in _offs:
            j = ctx_i[k] + _o
            if j >= len(common) or j >= ctx_i[k + 1]:
                continue
            a = common[j]; c2w, K = _cam(sd, COND_CAM, a)
            tgt.append({"gt": _rgb(sd, COND_CAM, a), "c2w": norm(c2w), "K": K,
                        "view": COND_CAM, "frame": int(a), "time": j / 2.0,
                        "offset": _o, "dyn_mask": None, "covis_mask": None})
    hw = cond[0]["image"].shape[:2]
    meta = {"scene": scene, "cond_cam": COND_CAM, "tgt_cam": COND_CAM, "prenormalized": True,
            "n_cond": len(cond), "n_tgt": len(tgt), "native_hw": hw, "dt": dt}
    return {"cond": cond, "targets": {"all": tgt}, "meta": meta}
