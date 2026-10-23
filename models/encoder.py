"""VGGT-based scene encoder shared by the frozen static tower and the trainable dynamic tower."""
import einops
import torch.nn as nn
import torch.nn.functional as F

from vggt.models.vggt import VGGT

VGGT_DIM = 1024
VGGT_PATCH = 14
VGGT_IMSIZE = 518
CAM_ENC_DIM = 11


def vggt_input_hw(h, w):
    """Resize rule of the encoder input: long side 518, short side rounded down to a multiple of 14."""
    if h > w:
        return VGGT_IMSIZE, (int(VGGT_IMSIZE * w / h) // VGGT_PATCH) * VGGT_PATCH
    return (int(VGGT_IMSIZE * h / w) // VGGT_PATCH) * VGGT_PATCH, VGGT_IMSIZE


class Encoder(nn.Module):
    def __init__(self, out_dim, depth=24, cross_attn_static=False, time_token=False, freeze_patch_embed=False):
        super().__init__()
        self.time_token = time_token
        self.vggt = VGGT(depth=depth, cross_attn_static=cross_attn_static, cross_attn_kv_dim=out_dim, time_token=time_token)
        if freeze_patch_embed:
            for p in self.vggt.aggregator.patch_embed.parameters():
                p.requires_grad_(False)
        self.camera_mlp = nn.Sequential(nn.Linear(CAM_ENC_DIM, VGGT_DIM), nn.SiLU(), nn.Linear(VGGT_DIM, VGGT_DIM))
        self.geo_feature_connector = nn.Linear(2 * VGGT_DIM, out_dim)
        self.geo_feature_norm = nn.LayerNorm(out_dim, bias=False)
        self.time_token_connector = nn.Linear(2 * VGGT_DIM, out_dim) if time_token else None

    def forward(self, images, cam_token, static_kv=None, return_tokens=False):
        """images [b,v,3,h,w], cam_token [b,v,11] -> feat [b,v,p,out_dim] (+ time token [b,v,out_dim], + DPT inputs)."""
        b, v = images.shape[:2]
        x = einops.rearrange(images, "b v c h w -> (b v) c h w")
        x = F.interpolate(x, size=vggt_input_hw(*x.shape[-2:]), mode="bilinear", antialias=True)
        x = einops.rearrange(x, "(b v) c h w -> b v c h w", b=b, v=v)
        cam = self.camera_mlp(cam_token).unsqueeze(2)
        out = self.vggt(x, cam, static_kv=static_kv, return_time_token=self.time_token, return_tokens=return_tokens)
        tokens = out[0] if isinstance(out, tuple) else out
        feat = self.geo_feature_norm(self.geo_feature_connector(tokens[:, :, self.vggt.aggregator.patch_start_idx:, :]))
        if not self.time_token:
            return feat
        time_tok = self.time_token_connector(out[1])
        if return_tokens:
            return feat, time_tok, (out[2], out[3], x)
        return feat, time_tok
