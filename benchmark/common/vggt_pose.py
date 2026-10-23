"""VGGT-1B camera estimation (extrinsics + intrinsics at native resolution) on letterboxed 518x518 inputs."""
import numpy as np
import torch
import torch.nn as nn

from vggt.heads.camera_head import CameraHead
from vggt.models.aggregator import Aggregator
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

RES = 518
_MODEL = None


class _VGGTCamera(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = Aggregator(img_size=RES, patch_size=14, embed_dim=1024)
        self.camera_head = CameraHead(dim_in=2 * 1024)


def _model(device):
    global _MODEL
    if _MODEL is None:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        sd = load_file(hf_hub_download("facebook/VGGT-1B", "model.safetensors"))
        m = _VGGTCamera()
        missing, _ = m.load_state_dict(sd, strict=False)
        assert not missing, f"VGGT-1B weights missing {len(missing)} keys, e.g. {missing[:3]}"
        _MODEL = m.to(device).eval()
    return _MODEL


@torch.no_grad()
def estimate(images, device="cuda"):
    """images: list of [H, W, 3] float RGB in [0, 1], all the same size -> (c2w [N,4,4], K [N,3,3]) in native pixels."""
    m = _model(device)
    t = torch.stack([torch.from_numpy(np.transpose(np.clip(i, 0, 1), (2, 0, 1))).float() for i in images]).to(device)
    H0, W0 = t.shape[-2:]
    sc = RES / max(H0, W0)
    nh, nw = int(round(H0 * sc)), int(round(W0 * sc))
    t = torch.nn.functional.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
    py, px = (RES - nh) // 2, (RES - nw) // 2
    x = torch.zeros(t.shape[0], 3, RES, RES, device=device)
    x[:, :, py:py + nh, px:px + nw] = t
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        toks, _ = m.aggregator(x[None])
    extr, intr = pose_encoding_to_extri_intri(m.camera_head(toks)[-1].float(), (RES, RES))
    e, K5 = extr.squeeze(0).cpu().numpy(), intr.squeeze(0).cpu().numpy()
    w2c = np.tile(np.eye(4, dtype=np.float32), (len(e), 1, 1))
    w2c[:, :3, :4] = e
    c2w = np.linalg.inv(w2c).astype(np.float32)
    K = K5.copy()
    K[:, 0, 0] /= sc
    K[:, 1, 1] /= sc
    K[:, 0, 2] = (K5[:, 0, 2] - px) / sc
    K[:, 1, 2] = (K5[:, 1, 2] - py) / sc
    return c2w, K.astype(np.float32)


def scene_cameras(cond, items, cond_images=None, device="cuda"):
    """Cameras for [cond | targets]. Targets that reuse a conditioning camera (static rigs) inherit its estimate."""
    cond_images = cond_images if cond_images is not None else [c["image"] for c in cond]
    cc = np.stack([np.asarray(c["c2w"], np.float64) for c in cond])
    idx = []
    for _, it in items:
        err = np.abs(cc - np.asarray(it["c2w"], np.float64)).max(axis=(1, 2))
        idx.append(int(err.argmin()) if err.min() <= 1e-4 else -1)
    if all(i >= 0 for i in idx):
        c2w, K = estimate(cond_images, device)
        ix = np.asarray(idx)
        return np.concatenate([c2w, c2w[ix]]), np.concatenate([K, K[ix]])
    return estimate(cond_images + [it["gt"] for _, it in items], device)
