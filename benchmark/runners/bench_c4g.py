#!/usr/bin/env python3
"""C4G runner (224^2 input, target-view pose TTO): protocol loaders -> render -> protocol scorer."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BENCH = Path(__file__).resolve().parent.parent   # benchmark/
for _d in ("common", "datasets", "runners"):
    sys.path.insert(0, str(BENCH / _d))
from bench_paths import BASELINES, OUT, dataset, weights  # noqa: E402

C4G = BASELINES / "C4G"
sys.path.insert(3, str(C4G))

CKPT = str(weights("c4g", "c4g_reconstruction.ckpt"))
RES = 224  # C4G square render size
NEAR, FAR = 0.1, 100.0
ALIGN_POSE = os.environ.get("C4G_ALIGN_POSE", "").lower() in ("1", "true", "yes")  # pose-only TTO (+evaluation=nvidia)
# Target-view pose TTO in the render path (port of C4G test_step_align).
C4G_TTO = os.environ.get("C4G_TTO", "1").lower() in ("1", "true", "yes")
C4G_TTO_STEPS = int(os.environ.get("C4G_TTO_STEPS", "2000"))
_LPIPS_TTO = None


def _tto_lpips():
    global _LPIPS_TTO
    if _LPIPS_TTO is None:
        import lpips as _lp
        _LPIPS_TTO = _lp.LPIPS(net="vgg").cuda().eval()
    return _LPIPS_TTO


def _c4g_tto(decoder, gaussians, extr, intr, near, far, gt_224, steps):
    """Optimize per-target camera poses against the GT targets at 224^2; returns refined extrinsics."""
    import torch.nn.functional as F
    v_ = extr.shape[1]
    rot_d = torch.zeros([1, v_, 3], requires_grad=True, device="cuda")
    trans_d = torch.zeros([1, v_, 3], requires_grad=True, device="cuda")
    opt = torch.optim.Adam([{"params": [rot_d], "lr": 0.005}, {"params": [trans_d], "lr": 0.005}])
    lpips_fn = _tto_lpips()
    extr_opt = extr.clone(); prev = None; pc = 0
    for it in range(steps):
        opt.zero_grad()
        # run_scene is under no_grad; re-enable grad for TTO
        with torch.enable_grad():
            out = decoder.forward(gaussians, extr_opt, intr, near, far, (RES, RES),
                                  cam_rot_delta=rot_d, cam_trans_delta=trans_d)
            pred = out.color
            loss = F.mse_loss(pred, gt_224) + lpips_fn(pred[0] * 2 - 1, gt_224[0] * 2 - 1).mean()
        loss.backward()
        with torch.no_grad():
            opt.step()
            new_w2c = update_pose(cam_trans_delta=trans_d.reshape(-1, 3),
                                  cam_rot_delta=rot_d.reshape(-1, 3),
                                  extrinsics=extr_opt.reshape(-1, 4, 4))
            rot_d.data.fill_(0); trans_d.data.fill_(0)
            extr_opt = new_w2c.reshape(1, v_, 4, 4)
        if prev is not None and abs(loss.item() - prev) < 1e-5:
            pc += 1
            if pc >= 10 and it >= 100:
                break
        else:
            pc = 0
        prev = loss.item()
    return extr_opt
OUT_ROOT = OUT

# RealEstate10K normalized intrinsics assumed by C4G.
RE10K = np.array([[0.4836, 0.0, 0.5],
                  [0.0, 0.8597, 0.5],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

os.chdir(str(C4G))
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from src.config import load_typed_root_config  # noqa: E402
from src.global_cfg import set_cfg  # noqa: E402
from src.model.encoder import get_encoder  # noqa: E402
from src.model.decoder import get_decoder  # noqa: E402
from src.misc.cam_utils import camera_normalization, update_pose  # noqa: E402
from src.dataset.shims.crop_shim import apply_crop_shim_to_views  # noqa: E402
from src.dataset.shims.normalize_shim import normalize_image  # noqa: E402
from protocol_score import Scorer  # noqa: E402

NV = str(dataset("NVIDIA"))


# dataset registry
def _kubric_specs():
    import bench_ds_kubric4d as m
    return dict(name="kubric4d", loader=m.load_kubric4d_scene,
                scenes=list(m.KUBRIC4D_SCENES), native_hw=(384, 576),
                groups=("all",), has_mask=True)


def _camxtime_specs():
    import bench_ds_camxtime as m
    return dict(name="camxtime", loader=m.load_camxtime_scene,
                scenes=list(m.list_test_scenes()), native_hw=(512, 512),
                groups=("all",), has_mask=False)


def _charge_specs():
    import bench_ds_charge as m
    return dict(name="charge", loader=m.load_charge_scene,
                scenes=list(m.SCENES), native_hw=(858, 2048),
                groups=("all",), has_mask=True)


def _n3dv_specs():
    import bench_ds_n3dv as m
    return dict(name="n3dv", loader=m.load_n3dv_scene,
                scenes=list(m.N3DV_SCENES), native_hw=(1014, 1352),
                groups=("all",), has_mask=False)


def _nvidia_specs():
    import bench_ds_nvidia as m
    return dict(name="nvidia", loader=m.load_nvidia, scenes=list(m.NVIDIA_SCENES),
                native_hw=(540, 960), groups=("all",), has_mask=False)


def _adt_specs():
    import bench_ds_adt as m
    return dict(name="adt", loader=m.load_adt_scene, scenes=list(m.ADT_SCENES),
                native_hw=(512, 512), groups=("all",), has_mask=False)


def _dycheck_specs():
    import bench_ds_dycheck as m
    return dict(name="dycheck", loader=m.load_dycheck_scene, scenes=list(m.DYCHECK_SCENES),
                native_hw=(960, 720), groups=("all",), has_mask=False)


def _tum_specs():
    import bench_ds_tum as m
    return dict(name="tum", loader=m.load_tum_scene, scenes=list(m.TUM_SCENES),
                native_hw=(480, 640), groups=("all",), has_mask=False)

REGISTRY = {"kubric4d": _kubric_specs, "camxtime": _camxtime_specs, "charge": _charge_specs, "n3dv": _n3dv_specs,
            "nvidia": _nvidia_specs, "adt": _adt_specs, "dycheck": _dycheck_specs, "tum": _tum_specs}


# model build
def build_cfg():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(C4G / "config"), version_base=None):
        cfg_dict = compose(
            config_name="main",
            overrides=[
                "+evaluation=nvidia",
                "mode=test",
                f"dataset.nvidia.roots=[{NV}]",
                "dataset.nvidia.target_cameras=[1,2,3,4,5,6,7,8,9,10,11,12]",
                "test.nvidia_view_mode=all12",
                f"test.align_pose={str(ALIGN_POSE).lower()}",
                f"checkpointing.load={CKPT}",
            ] + ([f"+model.encoder.normalize_type={os.environ['C4G_TIME_NORM']}"]
                 if os.environ.get("C4G_TIME_NORM") else []),
        )
    set_cfg(cfg_dict)
    return load_typed_root_config(cfg_dict)


def wait_for_gpu(min_free_gb=6.0, tries=40):
    for _ in range(tries):
        free, _tot = torch.cuda.mem_get_info()
        if free / 1e9 >= min_free_gb:
            return
        print(f"[gpu] only {free/1e9:.1f}GB free (<{min_free_gb}); wait 30s", flush=True)
        time.sleep(30)


def build_model(cfg):
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    sd = torch.load(CKPT, map_location="cpu")["state_dict"]
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    missing = [m for m in missing if not m.startswith(("backbone.backbone.",))]
    print(f"[ckpt] loaded encoder: {len(enc_sd)} keys; "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    wait_for_gpu(min_free_gb=6.0)
    encoder = encoder.cuda().eval()
    decoder = decoder.cuda().eval()
    return encoder, decoder


# geometry / framing helpers
C4G_RESIZE = os.environ.get("C4G_RESIZE", "1").lower() in ("1", "true", "yes")


# C4G_PP_SHIFT: widen the symmetric frustum to cover an off-centre principal point, then crop.
C4G_PP_SHIFT = os.environ.get("C4G_PP_SHIFT", "").lower() in ("1", "true", "yes")


def pp_expand_K_and_window(K_px, native_hw):
    """Return (centred normalized K for rendering, (x0,x1,y0,y1) crop window in [0,1] of the render)."""
    h, w = native_hw
    fx, fy, cx, cy = float(K_px[0, 0]), float(K_px[1, 1]), float(K_px[0, 2]), float(K_px[1, 2])
    Tx = max(cx, w - cx) / fx
    Ty = max(cy, h - cy) / fy
    Kr = np.eye(3, dtype=np.float32)
    Kr[0, 0] = 0.5 / Tx; Kr[0, 2] = 0.5
    Kr[1, 1] = 0.5 / Ty; Kr[1, 2] = 0.5
    x0 = (-cx / fx + Tx) / (2 * Tx); x1 = ((w - cx) / fx + Tx) / (2 * Tx)
    y0 = (-cy / fy + Ty) / (2 * Ty); y1 = ((h - cy) / fy + Ty) / (2 * Ty)
    return Kr, (x0, x1, y0, y1)


def crop_window(img_hwc, win):
    """Crop a [0,1]-normalized window (x0,x1,y0,y1) out of an HxWx3 image."""
    h, w = img_hwc.shape[:2]
    x0, x1, y0, y1 = win
    a, b = int(round(x0 * w)), int(round(x1 * w))
    c, d = int(round(y0 * h)), int(round(y1 * h))
    a = max(0, a); c = max(0, c); b = min(w, max(b, a + 1)); d = min(h, max(d, c + 1))
    return img_hwc[c:d, a:b]


# C4G_LETTERBOX: pad to square and render the whole frame.
C4G_LETTERBOX = os.environ.get("C4G_LETTERBOX", "").lower() in ("1", "true", "yes")


def letterbox_to_square(img_chw):
    """Zero-pad a [C,H,W] image to a square of side max(H,W). Returns (padded, pad_top, pad_left, S)."""
    import torch.nn.functional as _F
    c, h, w = img_chw.shape[-3:]
    S = max(h, w)
    pt, pl = (S - h) // 2, (S - w) // 2
    return _F.pad(img_chw, (pl, S - w - pl, pt, S - h - pt)), pt, pl, S


def norm_K_letterbox(K_px, native_hw):
    """Normalized K for the letterboxed square."""
    h, w = native_hw
    S = max(h, w); pt, pl = (S - h) // 2, (S - w) // 2
    K = np.asarray(K_px, dtype=np.float32).copy()
    K[0, 2] += pl; K[1, 2] += pt
    K[0, :] /= S; K[1, :] /= S
    return K


# C4G_RENDER_GT: render at GT resolution instead of 224^2.
C4G_RENDER_GT = os.environ.get("C4G_RENDER_GT", "").lower() in ("1", "true", "yes")


def gt_render_hw(native_hw):
    """GT resolution rounded to /8 for the rasterizer."""
    h, w = native_hw
    return max(8, int(round(h / 8)) * 8), max(8, int(round(w / 8)) * 8)


def render_hw_for(native_hw, base=RES):
    """Aspect-matched render size for the full frame (long side ~ base*W/H, /8 for the rasterizer)."""
    h, w = native_hw
    if w >= h:
        rw = int(round(base * w / h / 8)) * 8; rh = int(round(base / 8)) * 8
    else:
        rh = int(round(base * h / w / 8)) * 8; rw = int(round(base / 8)) * 8
    return max(8, rh), max(8, rw)


# Focal mismatch: C4G does not read K, so its geometry assumes the RE10K intrinsics.
C4G_RENDER_RE10K = os.environ.get("C4G_RENDER_RE10K", "").lower() in ("1", "true", "yes")
C4G_GAUSS_SQUASH = os.environ.get("C4G_GAUSS_SQUASH", "").lower() in ("1", "true", "yes")
# C4G_FOCAL_FIX: narrow the render focal to match the squashed scene.
C4G_FOCAL_FIX = os.environ.get("C4G_FOCAL_FIX", "1").lower() in ("1", "true", "yes")

# C4G_CAM_SQUASH: squash target camera centres along with the Gaussian means.
C4G_CAM_SQUASH = os.environ.get("C4G_CAM_SQUASH", "1").lower() in ("1", "true", "yes")
# C4G_FY_FIX: also scale fy under C4G_FOCAL_FIX.
C4G_FY_FIX = os.environ.get("C4G_FY_FIX", "").lower() in ("1", "true", "yes")

# C4G_ISO_FOCAL: isotropic render focal (fx := fy).
C4G_ISO_FOCAL = os.environ.get("C4G_ISO_FOCAL", "1").lower() in ("1", "true", "yes")
# C4G_SQUASH_ISO: use focal_ratio factors in squash_gaussians.
C4G_SQUASH_ISO = os.environ.get("C4G_SQUASH_ISO", "").lower() in ("1", "true", "yes")

# C4G_CAM_DX/DY/DZ: manual target-camera translation offset (cond[0] frame).
C4G_CAM_DX = float(os.environ.get("C4G_CAM_DX", "0.0"))
C4G_CAM_DY = float(os.environ.get("C4G_CAM_DY", "0.0"))
C4G_CAM_DZ = float(os.environ.get("C4G_CAM_DZ", "0.0"))


def focal_ratio(K_px, native_hw):
    """Squash factors for the Gaussian means AND the camera centres."""
    h, w = native_hw
    fxn, fyn = float(K_px[0][0]) / w, float(K_px[1][1]) / h
    if C4G_ISO_FOCAL:
        return fxn / fyn, 1.0
    sx = float(RE10K[0, 0]) / fxn
    sy = float(RE10K[1, 1]) / fyn
    return sx, sy


# C4G_PP_ROT: rotate target cameras so the symmetric frustum is centred on the true principal ray.
C4G_PP_ROT = os.environ.get("C4G_PP_ROT", "1").lower() in ("1", "true", "yes")


def pp_rotation(K_px, native_hw):
    """4x4 camera-frame rotation putting +z along the true ray through the image centre."""
    h, w = native_hw
    fx, fy, cx, cy = float(K_px[0][0]), float(K_px[1][1]), float(K_px[0][2]), float(K_px[1][2])
    if C4G_ISO_FOCAL:
        fyn = fy / h
        d = np.array([((w / 2 - cx) / w) / fyn, ((h / 2 - cy) / h) / fyn, 1.0], dtype=np.float64)
    else:
        _sx, _sy = focal_ratio(K_px, native_hw) if C4G_FOCAL_FIX else (1.0, 1.0)
        d = np.array([(w / 2 - cx) / fx * _sx, (h / 2 - cy) / fy * _sy, 1.0], dtype=np.float64)
    d /= np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d); c = float(np.dot(z, d))
    if np.linalg.norm(v) < 1e-12:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    T = np.eye(4, dtype=np.float32); T[:3, :3] = R
    return T


C4G_FX_MULT = float(os.environ.get("C4G_FX_MULT", "1.0"))
C4G_FY_MULT = float(os.environ.get("C4G_FY_MULT", "1.0"))
# C4G_RE10K_NOCROP: render with the RE10K K and do not crop.
C4G_RE10K_NOCROP = os.environ.get("C4G_RE10K_NOCROP", "").lower() in ("1", "true", "yes")


def re10k_window(K_px, native_hw):
    """True-K view as a [0,1] sub-window of a render made with the wider RE10K frustum."""
    h, w = native_hw
    fxn, fyn = float(K_px[0][0]) / w, float(K_px[1][1]) / h
    Tx_true, Ty_true = 0.5 / fxn, 0.5 / fyn          # half-extent (tan) of the real camera
    Tx_re, Ty_re = 0.5 / RE10K[0, 0], 0.5 / RE10K[1, 1]
    x0, x1 = 0.5 - Tx_true / (2 * Tx_re), 0.5 + Tx_true / (2 * Tx_re)
    y0, y1 = 0.5 - Ty_true / (2 * Ty_re), 0.5 + Ty_true / (2 * Ty_re)
    return (max(0.0, x0), min(1.0, x1), max(0.0, y0), min(1.0, y1))


def squash_gaussians(g, K_px, native_hw):
    """Squash Gaussians laterally by f_assumed/f_true so the scene matches the real K."""
    if C4G_SQUASH_ISO:
        sx, sy = focal_ratio(K_px, native_hw)
    else:
        h, w = native_hw
        sx = float(RE10K[0, 0]) / (float(K_px[0][0]) / w)
        sy = float(RE10K[1, 1]) / (float(K_px[1][1]) / h)
    import copy
    g2 = copy.copy(g)
    m = g.means.clone(); m[..., 0] *= sx; m[..., 1] *= sy
    g2.means = m
    if getattr(g, "scales", None) is not None:
        sc = g.scales.clone(); sc[..., 0] *= sx; sc[..., 1] *= sy; g2.scales = sc
    if getattr(g, "covariances", None) is not None:
        S = torch.diag(torch.tensor([sx, sy, 1.0], device=g.covariances.device, dtype=g.covariances.dtype))
        g2.covariances = S @ g.covariances @ S
    return g2

def norm_K_fullframe(K_px, native_hw):
    """Real pixel K -> [0,1]-normalized by the view's own size. Resize-invariant."""
    h_in, w_in = native_hw
    K = np.asarray(K_px, dtype=np.float32).copy()
    K[0, :] /= w_in
    K[1, :] /= h_in
    return K


def crop_adjusted_re10k(native_hw):
    """RE10K normalized intrinsics adjusted for rescale+center-crop to 224^2."""
    h_in, w_in = native_hw
    sf = max(RES / h_in, RES / w_in)
    h_s, w_s = round(h_in * sf), round(w_in * sf)
    K = RE10K.copy()
    K[0, 0] *= w_s / RES
    K[1, 1] *= h_s / RES
    return K


def crop_adjusted_norm_K(K_px, native_hw):
    """Pixel K normalized by the image size, then adjusted for the crop shim to 224^2."""
    h_in, w_in = native_hw
    K = np.asarray(K_px, dtype=np.float32).copy()
    K[0, :] /= w_in   # fx, cx  -> fx/W, cx/W
    K[1, :] /= h_in   # fy, cy  -> fy/H, cy/H
    sf = max(RES / h_in, RES / w_in)
    h_s, w_s = round(h_in * sf), round(w_in * sf)
    K[0, 0] *= w_s / RES
    K[1, 1] *= h_s / RES
    return K


def _covis_masked(pred, gt, covis_bool, scorer):
    """Covisibility-masked PSNR/SSIM/LPIPS at the pred resolution (adapted from MoVieS eval.compute_masked)."""
    import protocol_score as _PS
    from skimage.metrics import structural_similarity
    ph, pw = pred.shape[:2]
    gt_r = np.clip(_PS.resize_to(np.clip(gt, 0, 1).astype(np.float32), (ph, pw)), 0, 1).astype(np.float32)
    m = _PS.resize_mask_to(covis_bool.astype(np.float32), (ph, pw)).astype(np.float32)   # nearest
    msum = float(max(m.sum(), 1.0))
    diff = ((gt_r - pred) ** 2).mean(2)                                   # [h,w]
    mse = float((diff * m).sum() / msum)
    psnr = -10.0 * np.log10(max(mse, 1e-10))
    _, smap = structural_similarity(gt_r, pred, win_size=11, gaussian_weights=True,
                                    channel_axis=-1, data_range=1.0, full=True)
    ssim = float((smap.mean(2) * m).sum() / msum)
    lp = scorer.lpips_val((pred * m[..., None]).astype(np.float32), (gt_r * m[..., None]).astype(np.float32))
    return {"psnr": float(psnr), "ssim": ssim, "lpips": float(lp), "covis_frac": float(m.mean())}


def _resize_hwc(img_hwc, hw):
    """Bilinear resize an HxWx3 float image to (H,W)."""
    import torch.nn.functional as _F
    t = torch.from_numpy(np.ascontiguousarray(img_hwc)).permute(2, 0, 1)[None].float()
    return _F.interpolate(t, size=tuple(hw), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()


def center_square(img_hwc):
    """Center-square crop of an HxWx.. array (crop = min(H,W))."""
    h, w = img_hwc.shape[:2]
    c = min(h, w)
    r0, c0 = (h - c) // 2, (w - c) // 2
    return img_hwc[r0:r0 + c, c0:c0 + c]


def to_chw(img_hwc):
    return torch.from_numpy(np.ascontiguousarray(img_hwc)).permute(2, 0, 1).float()


def to_hwc(chw):
    return np.clip(np.transpose(chw, (1, 2, 0)), 0, 1)


# per-scene evaluation
def _rot90_view(rec, key, k):
    """Rotate image k*90 CCW and carry K + c2w with it (roll about the optical axis)."""
    import numpy as _np
    img = _np.asarray(rec[key])
    H, W = img.shape[:2]
    rec[key] = _np.ascontiguousarray(_np.rot90(img, k))
    K = _np.asarray(rec["K"], _np.float64).copy()
    R = _np.eye(3)
    for _ in range(k % 4):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        K[0, 0], K[1, 1] = fy, fx
        K[0, 2], K[1, 2] = cy, (W - 1) - cx
        H, W = W, H
        R = _np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ R
    rec["K"] = K.astype(_np.float32)
    c2w = _np.asarray(rec["c2w"], _np.float64).copy()
    c2w[:3, :3] = c2w[:3, :3] @ R.T
    rec["c2w"] = c2w.astype(_np.float32)
    return rec


def _rot90_data(data, k, spec):
    for c in data["cond"]:
        _rot90_view(c, "image", k)
    for g in data["targets"]:
        for t in data["targets"][g]:
            _rot90_view(t, "gt", k)
            if t.get("dyn_mask") is not None:
                import numpy as _np
                t["dyn_mask"] = _np.ascontiguousarray(_np.rot90(_np.asarray(t["dyn_mask"]), k))
    if k % 2:
        h, w = spec["native_hw"]
        spec["native_hw"] = (w, h)
    return data


@torch.no_grad()
def run_scene(spec, scene, encoder, decoder, scorer, out_dir):
    data = spec["loader"](scene)
    _k = int(os.environ.get("C4G_ROT90", "0"))
    if _k:
        data = _rot90_data(data, _k, spec)
    _mx = int(os.environ.get("C4G_MAX_CTX", "0"))
    if _mx and len(data["cond"]) > _mx:
        _idx = np.linspace(0, len(data["cond"]) - 1, _mx).round().astype(int)
        data["cond"] = [data["cond"][i] for i in sorted(set(_idx.tolist()))]
    cond = data["cond"]
    tgt_groups = data["targets"]
    # C4G_SELFRECON: use the conditioning views as targets.
    if os.environ.get("C4G_SELFRECON", "").lower() in ("1", "true", "yes"):
        _self = []
        for _c in cond:
            _t = dict(_c)
            _t["gt"] = _c["image"] if "image" in _c else _c["gt"]
            _self.append(_t)
        tgt_groups = {"all": _self}
        print(f"[selfrecon] targets replaced by {len(_self)} cond views", flush=True)
    V = len(cond)

    # cond[i] is at logical time i
    use_time = all("time" in c for c in cond)
    frame2tidx = {int(cond[i]["frame"]): i for i in range(V)}
    def _tidx(it):
        return float(it["time"]) if use_time else frame2tidx[int(it["frame"])]

    # Joint cond+target extrinsic normalization (make_baseline_1 + relative).
    ctx_extr = torch.from_numpy(np.stack([np.asarray(c["c2w"], np.float32) for c in cond])).float()
    flat = []  # (group, tidx, gt_native, dyn_mask_native_or_None, cam, frame, K, covis, offset)
    tgt_extr_list = []
    for g in spec["groups"]:
        for t in tgt_groups[g]:
            tgt_extr_list.append(np.asarray(t["c2w"], np.float32))
            flat.append((g, _tidx(t), t["gt"],
                         t.get("dyn_mask"), t.get("cam", t.get("view")), int(t["frame"]),
                         t.get("K"), t.get("covis_mask"), t.get("offset")))
    tgt_extr = torch.from_numpy(np.stack(tgt_extr_list)).float()
    all_extr = torch.cat([ctx_extr, tgt_extr], dim=0)

    # prenormalized loaders use scale=1
    prenorm = bool(data.get("meta", {}).get("prenormalized"))
    origin = all_extr[0, :3, 3]
    cond_baseline = (ctx_extr[1:, :3, 3] - origin).norm(dim=1).max().item() if V > 1 else 0.0
    # BENCH_NORM_REF: apply make_baseline_1 even when prenormalized.
    if prenorm and os.environ.get("BENCH_NORM_REF") != "1":
        scale = 1.0
        scale_mode = "prenormalized (scene.json)"
    elif cond_baseline > 1e-6:
        scale = cond_baseline
        scale_mode = "cond-baseline (make_baseline_1)"
    else:  # static-cond fallback: span of all cams from cond[0]
        span = (all_extr[:, :3, 3] - origin).norm(dim=1).max().item()
        if span > 1e-6:
            scale = span
            scale_mode = "cond+target span (static-cond fallback)"
        else:
            scale = 1.0
            scale_mode = "fully-static rig (no normalisation)"
    all_extr = all_extr.clone()
    # C4G_POSE_SCALE_MULT: extra multiplier on the pose scale.
    scale *= float(os.environ.get("C4G_POSE_SCALE_MULT", "1.0"))
    all_extr[:, :3, 3] /= scale
    all_extr = camera_normalization(all_extr[0:1], all_extr)
    ctx_extr_n = all_extr[:V]
    tgt_extr_n = all_extr[V:]

    # Context images -> 224^2
    ctx_img = torch.stack([to_chw(c["image"]) for c in cond])           # [V,3,H,W]
    if C4G_LETTERBOX:
        import torch.nn.functional as _Fr
        _pad, _pt, _pl, _S = letterbox_to_square(ctx_img)          # aspect preserved, full FOV kept
        ctx_img_224 = _Fr.interpolate(_pad, size=(RES, RES), mode="bilinear", align_corners=False)
        K_adj = torch.from_numpy(norm_K_letterbox(cond[0]["K"], spec["native_hw"])).float()
    elif C4G_RESIZE:
        import torch.nn.functional as _Fr
        ctx_img_224 = _Fr.interpolate(ctx_img, size=(RES, RES), mode="bilinear", align_corners=False)
        K_adj = (torch.from_numpy(RE10K).float() if C4G_RENDER_RE10K
                 else torch.from_numpy(norm_K_fullframe(cond[0]["K"], spec["native_hw"])).float())
    else:
        K_native = torch.from_numpy(RE10K).unsqueeze(0).repeat(V, 1, 1)
        cropped = apply_crop_shim_to_views({"image": ctx_img, "intrinsics": K_native}, (RES, RES))
        ctx_img_224 = cropped["image"]                                   # [V,3,224,224]
        K_adj = torch.from_numpy(crop_adjusted_re10k(spec["native_hw"])).float()

    # Context intrinsics
    if C4G_RESIZE and C4G_RENDER_RE10K:
        ctx_intr = torch.from_numpy(RE10K).float().unsqueeze(0).repeat(1, V, 1, 1)
    elif C4G_LETTERBOX and all(c.get("K") is not None for c in cond):
        ctx_K = np.stack([norm_K_letterbox(np.asarray(c["K"]), c["image"].shape[:2]) for c in cond])
        ctx_intr = torch.from_numpy(ctx_K).float().unsqueeze(0)
    elif C4G_RESIZE and all(c.get("K") is not None for c in cond):
        _f = (lambda K, hw: pp_expand_K_and_window(K, hw)[0]) if C4G_PP_SHIFT else norm_K_fullframe
        ctx_K = np.stack([_f(np.asarray(c["K"]), c["image"].shape[:2]) for c in cond])
        ctx_intr = torch.from_numpy(ctx_K).float().unsqueeze(0)
    elif os.environ.get("C4G_CTX_FIX") and all(c.get("K") is not None for c in cond):
        ctx_K = np.stack([crop_adjusted_norm_K(np.asarray(c["K"]), c["image"].shape[:2]) for c in cond])
        ctx_intr = torch.from_numpy(ctx_K).float().unsqueeze(0)  # [1,V,3,3]
    else:
        ctx_intr = K_adj.unsqueeze(0).repeat(1, V, 1, 1)  # fixed RE10K

    context = {
        "image": normalize_image(ctx_img_224).unsqueeze(0).cuda(),
        "index": torch.arange(V, dtype=torch.int64).unsqueeze(0).cuda(),
        "camera": torch.arange(V, dtype=torch.int64).unsqueeze(0).cuda(),
        "extrinsics": ctx_extr_n.unsqueeze(0).cuda(),
        "intrinsics": ctx_intr.cuda(),
        "near": torch.full((1, V), NEAR / scale).cuda(),
        "far": torch.full((1, V), FAR / scale).cuda(),
    }

    unique_ts, _ts_inv = torch.unique(torch.tensor([f[1] for f in flat]), return_inverse=True)
    gaussians_per_ts = encoder(context, 0, target_timestamps=unique_ts.cuda())

    if os.environ.get("C4G_DEBUG"):
        g0 = gaussians_per_ts[list(gaussians_per_ts)[0]]
        gm = g0.means[0].reshape(-1, 3)
        cc = ctx_extr_n[:, :3, 3]; tc = tgt_extr_n[:, :3, 3]
        print(f"[DBG {scene}] scale={scale:.4f} near={NEAR/scale:.3f} far={FAR/scale:.3f}", flush=True)
        print(f"[DBG {scene}] gauss means: min={gm.min(0).values.cpu().numpy().round(2)} "
              f"max={gm.max(0).values.cpu().numpy().round(2)} mean={gm.mean(0).cpu().numpy().round(2)}", flush=True)
        print(f"[DBG {scene}] ctx cam centers bbox: min={cc.min(0).values.numpy().round(2)} max={cc.max(0).values.numpy().round(2)}", flush=True)
        print(f"[DBG {scene}] tgt cam centers bbox: min={tc.min(0).values.numpy().round(2)} max={tc.max(0).values.numpy().round(2)}", flush=True)
        # depth of gaussians in the first target camera frame (z along view dir)
        w2c = torch.inverse(tgt_extr_n[0])
        pc = (w2c[:3, :3] @ gm.cpu().T).T + w2c[:3, 3]
        print(f"[DBG {scene}] gauss depth in tgt0 cam: z min={pc[:,2].min():.2f} max={pc[:,2].max():.2f} "
              f"median={pc[:,2].median():.2f}  (need in [{NEAR/scale:.2f},{FAR/scale:.2f}], z>0 = in front)", flush=True)

    sdir = out_dir / scene
    (sdir / "pred").mkdir(parents=True, exist_ok=True)
    rows = []
    preds_cache = []  # (group, pred_hwc, gt_crop, tag)
    fix_intr = bool(os.environ.get("C4G_FIX")) or C4G_RESIZE or C4G_LETTERBOX
    for ti, t in enumerate(unique_ts.tolist()):
        # group by unique index rather than float equality
        idxs = [i for i in range(len(flat)) if int(_ts_inv[i]) == ti]
        extr = tgt_extr_n[idxs].unsqueeze(0).cuda()
        if C4G_CAM_SQUASH:  # squash camera centres with the scene
            _e = extr[0].clone()
            for _j, _i in enumerate(idxs):
                _Kj = flat[_i][6]
                if _Kj is None:
                    continue
                _sx, _sy = focal_ratio(_Kj, flat[_i][2].shape[:2])
                _e[_j, 0, 3] *= _sx
                _e[_j, 1, 3] *= _sy
            extr = _e.unsqueeze(0)
        if C4G_CAM_DX or C4G_CAM_DY or C4G_CAM_DZ:      # manual camera move (diagnostic)
            _e = extr[0].clone()
            _e[:, 0, 3] += C4G_CAM_DX
            _e[:, 1, 3] += C4G_CAM_DY
            _e[:, 2, 3] += C4G_CAM_DZ
            extr = _e.unsqueeze(0)
        if C4G_PP_ROT:                     # compensate the off-centre principal via the extrinsic
            _e = extr[0].clone()
            for _j, _i in enumerate(idxs):
                _Kj = flat[_i][6]
                if _Kj is None:
                    continue
                _T = torch.from_numpy(pp_rotation(_Kj, flat[_i][2].shape[:2])).to(_e.device, _e.dtype)
                _e[_j] = _e[_j] @ _T
            extr = _e.unsqueeze(0)
        if fix_intr:
            per_tgt = []
            for i in idxs:
                Ki = flat[i][6]
                gh, gw = flat[i][2].shape[:2]
                per_tgt.append(K_adj.numpy() if Ki is None
                               else norm_K_fullframe(Ki, (gh, gw)) if C4G_LETTERBOX
                                     else (pp_expand_K_and_window(Ki, (gh, gw))[0] if (C4G_RESIZE and C4G_PP_SHIFT)
                                     else RE10K if (C4G_RESIZE and C4G_RENDER_RE10K)
                                     else norm_K_fullframe(Ki, (gh, gw)) if C4G_RESIZE
                                     else crop_adjusted_norm_K(Ki, (gh, gw))))
            _pt = np.stack(per_tgt)
            if C4G_ISO_FOCAL:                      # fx_n := fy_n  (isotropic on the square render)
                _pt[:, 0, 0] = _pt[:, 1, 1]
            elif C4G_FOCAL_FIX:                    # narrow the frustum by 1/s to match the squash
                for _j, _i in enumerate(idxs):
                    _Kj = flat[_i][6]
                    if _Kj is None: continue
                    _sx, _sy = focal_ratio(_Kj, flat[_i][2].shape[:2])
                    _pt[_j, 0, 0] /= _sx
                    if C4G_FY_FIX:
                        _pt[_j, 1, 1] /= _sy
            if C4G_FX_MULT != 1.0: _pt[:, 0, 0] *= C4G_FX_MULT
            if C4G_FY_MULT != 1.0: _pt[:, 1, 1] *= C4G_FY_MULT
            intr = torch.from_numpy(_pt).float().unsqueeze(0).cuda()
        else:
            intr = K_adj.unsqueeze(0).repeat(1, len(idxs), 1, 1).cuda()
        near = torch.full((1, len(idxs)), NEAR / scale).cuda()
        far = torch.full((1, len(idxs)), FAR / scale).cuda()
        # Group targets by K, squash per group, and run TTO within each group.
        _rhw = ((RES, RES) if os.environ.get("C4G_SQUARE") == "1" else
                gt_render_hw(spec["native_hw"]) if C4G_RENDER_GT
                else render_hw_for(spec["native_hw"]) if C4G_LETTERBOX else (RES, RES))
        _g0 = gaussians_per_ts[t]
        _do_squash = C4G_GAUSS_SQUASH or C4G_FOCAL_FIX
        groups = {}
        for _j, _i in enumerate(idxs):
            _Kj = flat[_i][6]
            key = None if (_Kj is None or not _do_squash) else np.asarray(_Kj, np.float64).tobytes()
            groups.setdefault(key, []).append(_j)

        gt_224 = None
        if C4G_TTO:
            import torch.nn.functional as _F
            gt_sq = np.stack([(np.clip(flat[i][2], 0, 1).astype(np.float32) if C4G_RESIZE
                               else center_square(np.clip(flat[i][2], 0, 1).astype(np.float32))) for i in idxs])
            gt_224 = _F.interpolate(torch.from_numpy(gt_sq).permute(0, 3, 1, 2).float().cuda(),
                                    size=(RES, RES), mode="bilinear", align_corners=False).unsqueeze(0)

        colors = None
        for key, js in groups.items():
            _jt = torch.as_tensor(js, device=extr.device)
            _gg = _g0
            if key is not None:
                _gg = squash_gaussians(_g0, flat[idxs[js[0]]][6], flat[idxs[js[0]]][2].shape[:2])
            _e = extr[:, _jt]
            if C4G_TTO:
                _e = _c4g_tto(decoder, _gg, _e, intr[:, _jt], near[:, _jt], far[:, _jt],
                              gt_224[:, _jt], C4G_TTO_STEPS)
            _o = decoder.forward(_gg, _e, intr[:, _jt], near[:, _jt], far[:, _jt], _rhw)
            _c = _o.color[0].float().cpu().numpy()
            if colors is None:
                colors = np.zeros((len(idxs),) + _c.shape[1:], dtype=_c.dtype)
            colors[js] = _c
        for k, i in enumerate(idxs):
            g, tidx, gt_native, dyn, cam, frame, _K, covis, offset = flat[i]
            pred = to_hwc(colors[k])
            if os.environ.get("C4G_SAVE_RAW"):  # raw render before crop/resize
                _rd = sdir / "pred_raw"; _rd.mkdir(parents=True, exist_ok=True)
                Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8)).save(
                    _rd / f"{g}_{cam}_i{i:04d}_f{frame}.png")
            if C4G_RESIZE or C4G_LETTERBOX:
                # full-frame render -> upsample to native GT
                gt = np.clip(gt_native, 0, 1).astype(np.float32)
                if C4G_RESIZE and C4G_RENDER_RE10K and not C4G_RE10K_NOCROP and _K is not None:
                    # rendered through the wider RE10K frustum -> crop the true-K sub-window
                    pred = crop_window(pred, re10k_window(_K, gt.shape[:2]))
                elif C4G_RESIZE and C4G_PP_SHIFT and _K is not None:
                    # rendered frustum is symmetric+widened; crop the true GT window out of it
                    _, _win = pp_expand_K_and_window(np.asarray(_K), gt.shape[:2])
                    pred = crop_window(pred, _win)
                pred = _resize_hwc(pred, gt.shape[:2])
                _cv, _dyn = covis, dyn
            else:
                gt = center_square(np.clip(gt_native, 0, 1).astype(np.float32))
                _cv = center_square(covis) if covis is not None else None
                _dyn = center_square(dyn) if dyn is not None else None
            if covis is not None:                        # DyCheck covisibility-masked metric
                m = _covis_masked(pred, gt, _cv, scorer)
            else:
                mask = _dyn
                m = scorer.score(pred, gt, dyn_mask=mask)
            m.update(group=g, tidx=int(tidx), cam=str(cam), frame=int(frame))
            if offset is not None:
                m["offset"] = int(offset)
            rows.append(m)
            tag = f"{g}_{cam}_i{i:04d}_f{frame}"  # i = unique flat target index
            Image.fromarray((pred * 255).astype(np.uint8)).save(sdir / "pred" / f"{tag}.png")
            if len(preds_cache) < 512:
                preds_cache.append((g, pred, gt, tag))

    # ---- per-scene aggregation ----
    def agg(subset, keys):
        d = {}
        for k in keys:
            vals = [r[k] for r in subset if k in r and np.isfinite(r[k])]
            d[k] = float(np.mean(vals)) if vals else float("nan")
        d["n"] = len(subset)
        return d

    keys = ["psnr", "ssim", "lpips"] + (["dyn_psnr"] if spec["has_mask"] else [])
    per_scene = {"overall": agg(rows, keys),
                 "groups": {g: agg([r for r in rows if r["group"] == g], keys)
                            for g in spec["groups"]}}
    json.dump({"scene": scene, **per_scene, "rows": rows},
              open(sdir / "metrics.json", "w"), indent=2)

    # vis grid: GT / pred / |err|
    if preds_cache:
        pick = np.linspace(0, len(preds_cache) - 1, min(8, len(preds_cache))).round().astype(int)
        cols = []
        for i in pick:
            _g, pred, gt, _tag = preds_cache[i]
            gh, gw = gt.shape[:2]
            pr = np.asarray(Image.fromarray((pred * 255).astype(np.uint8))
                            .resize((gw, gh), Image.LANCZOS), np.float32) / 255.0
            err = np.abs(np.clip(gt, 0, 1) - np.clip(pr, 0, 1)).mean(axis=2)
            err = np.clip(err * 3.0, 0, 1)                      # 3x gain so the residual is legible
            err = np.stack([err, err, err], axis=2)
            cols.append(np.concatenate([gt, pr, err], axis=0))
        grid = (np.clip(np.concatenate(cols, axis=1), 0, 1) * 255).astype(np.uint8)
        # save the grid next to the per-scene dirs
        Image.fromarray(grid).save(sdir.parent / f"{scene}.png")
        _gd = os.environ.get("BENCH_GRID_DIR")
        if _gd:
            os.makedirs(_gd, exist_ok=True)
            _t = os.environ.get("BENCH_GRID_TAG", "")
            Image.fromarray(grid).save(os.path.join(_gd, f"{_t}__{scene}.png" if _t else f"{scene}.png"))

    return per_scene, scale_mode


# dataset driver
def run_dataset(ds_name, encoder, decoder, scorer, limit=None):
    spec = REGISTRY[ds_name]()
    scenes = spec["scenes"][:limit] if limit else spec["scenes"]
    _only = os.environ.get("C4G_SCENES")  # comma-separated scene subset
    if _only:
        _want = {x.strip() for x in _only.split(",") if x.strip()}
        scenes = [sc for sc in scenes if str(sc) in _want]
    out_dir = OUT_ROOT / ds_name / os.environ.get("C4G_OUT_NAME", "c4g")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n===== C4G / {ds_name}: {len(scenes)} scenes -> {out_dir} =====", flush=True)

    per_scene_all = {}
    scale_modes = set()
    for sc in scenes:
        attempt = 0
        while True:
            attempt += 1
            wait_for_gpu()
            try:
                ps, smode = run_scene(spec, sc, encoder, decoder, scorer, out_dir)
                per_scene_all[sc] = ps
                scale_modes.add(smode)
                gline = "  ".join(
                    f"{g}:PSNR {ps['groups'][g]['psnr']:.3f}"
                    + (f"/dyn {ps['groups'][g]['dyn_psnr']:.3f}"
                       if spec["has_mask"] and np.isfinite(ps['groups'][g].get('dyn_psnr', float('nan'))) else "")
                    for g in spec["groups"])
                print(f"[{ds_name}] {sc}: overall PSNR {ps['overall']['psnr']:.3f}  {gline}", flush=True)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and attempt <= 4:
                    print(f"[{ds_name}] {sc}: OOM (attempt {attempt}); empty_cache + wait 30s", flush=True)
                    torch.cuda.empty_cache()
                    time.sleep(30)
                    continue
                print(f"[{ds_name}] {sc}: FAILED {e}", flush=True)
                break

    # ---- across-scene aggregation (mean of per-scene means) ----
    keys = ["psnr", "ssim", "lpips"] + (["dyn_psnr"] if spec["has_mask"] else [])

    def across(selector):
        d = {}
        for k in keys:
            vals = [selector(ps)[k] for ps in per_scene_all.values()
                    if k in selector(ps) and np.isfinite(selector(ps)[k])]
            d[k] = float(np.mean(vals)) if vals else float("nan")
        return d

    summary = {
        "model": "C4G",
        "dataset": ds_name,
        "protocol": "eval_protocol.md (feed-forward NVS)",
        "framing": "center-square (C4G square-input; scored on input-covered FoV)",
        "input": f"{len(spec['scenes']) and 8} cond, C4G center-crop to 224^2 (VGGT native)",
        "render_res": "224^2 (square), upsampled to center-square native for scoring",
        "intrinsics": "fixed RE10K (crop-shim adjusted); GT target pose to render",
        "pose_norm": f"make_baseline_1 + relative-to-cond[0]; modes={sorted(scale_modes)}",
        "align_pose": ALIGN_POSE,
        "lpips": "vgg",
        "overall": across(lambda ps: ps["overall"]),
        "groups": {g: across(lambda ps, g=g: ps["groups"][g]) for g in spec["groups"]},
        "per_scene": per_scene_all,
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)

    print(f"\n----- {ds_name} SUMMARY (n={len(per_scene_all)} scenes) -----", flush=True)
    print(f"  overall: { {k: round(v, 4) for k, v in summary['overall'].items()} }", flush=True)
    for g in spec["groups"]:
        print(f"  {g}: { {k: round(v, 4) for k, v in summary['groups'][g].items()} }", flush=True)
    print(f"  saved -> {out_dir / 'summary.json'}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["all"] + list(REGISTRY))
    ap.add_argument("--limit", type=int, default=None, help="first N scenes (validation)")
    args = ap.parse_args()

    cfg = build_cfg()
    encoder, decoder = build_model(cfg)
    scorer = Scorer(device="cuda")

    order = ["kubric4d", "camxtime", "charge"] if args.dataset == "all" else [args.dataset]
    finals = {}
    for ds in order:
        finals[ds] = run_dataset(ds, encoder, decoder, scorer, limit=args.limit)

    print("\n########## ALL DATASETS DONE ##########", flush=True)
    for ds, s in finals.items():
        print(f"[{ds}] overall={ {k: round(v,3) for k,v in s['overall'].items()} }", flush=True)
        for g, gm in s["groups"].items():
            print(f"    {g}={ {k: round(v,3) for k,v in gm.items()} }", flush=True)


if __name__ == "__main__":
    main()
