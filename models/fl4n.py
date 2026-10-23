"""FL4N: frozen static encoder + trainable dynamic encoder fused by a zero-init gate into a frozen renderer."""
import re

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder, VGGT_DIM
from models.renderer import Renderer
from models.time_module import TimeModule
from vggt.heads.dpt_head import DPTHead


def interp_time_token(time_tok, cond_tau, target_tau):
    """Per-target conditioning = linear interpolation of the two cond time tokens bracketing its time."""
    b, K, d = time_tok.shape
    order = cond_tau.argsort(dim=1)
    cta = torch.gather(cond_tau, 1, order)
    tok = torch.gather(time_tok, 1, order[:, :, None].expand(-1, -1, d))
    hi = torch.searchsorted(cta.contiguous(), target_tau.contiguous()).clamp(1, K - 1)
    lo = hi - 1
    t_lo, t_hi = torch.gather(cta, 1, lo), torch.gather(cta, 1, hi)
    w = ((target_tau - t_lo) / (t_hi - t_lo).clamp(min=1e-6)).clamp(0.0, 1.0)
    tok_lo = torch.gather(tok, 1, lo[:, :, None].expand(-1, -1, d))
    tok_hi = torch.gather(tok, 1, hi[:, :, None].expand(-1, -1, d))
    return einops.rearrange((1.0 - w).unsqueeze(-1) * tok_lo + w.unsqueeze(-1) * tok_hi, "b v d -> (b v) d")


def uniform_sample_blocks(sd, dyn_depth):
    """Warm-start a depth-D aggregator from a deeper one by uniformly sampling its frame/global blocks."""
    pat = re.compile(r"(.*\.(?:frame_blocks|global_blocks)\.)(\d+)(\..*)")
    src_depth = max((int(pat.match(k).group(2)) for k in sd if pat.match(k)), default=-1) + 1
    idx = torch.linspace(0, src_depth - 1, dyn_depth).round().long().tolist()
    src_to_dst = {src: dst for dst, src in enumerate(idx)}
    out = {}
    for k, v in sd.items():
        m = pat.match(k)
        if m is None:
            out[k] = v
        elif int(m.group(2)) in src_to_dst:
            out[f"{m.group(1)}{src_to_dst[int(m.group(2))]}{m.group(3)}"] = v
    return out


class FL4N(nn.Module):
    def __init__(self, depth=12, hidden_size=768, patch_size=8, num_heads=12, static_depth=24, dyn_depth=8,
                 time_depth=2, freeze_patch_embed=False, pretrained_path=None, dyn_warm_start=True):
        super().__init__()
        self.gate = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.static_enc = Encoder(hidden_size, depth=static_depth)
        self.dyn_enc = Encoder(hidden_size, depth=dyn_depth, cross_attn_static=True, time_token=True,
                               freeze_patch_embed=freeze_patch_embed)
        self.time_module = TimeModule(hidden_size, num_heads, depth=time_depth)
        taps = [max(0, round((i + 1) * dyn_depth / 4) - 1) for i in range(4)]
        self.mask_head = DPTHead(dim_in=2 * VGGT_DIM, output_dim=1, activation="linear", no_conf=True,
                                 intermediate_layer_idx=taps)
        self.renderer = Renderer(depth, hidden_size, patch_size, num_heads)
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path, dyn_warm_start)
        for p in list(self.static_enc.parameters()) + list(self.renderer.parameters()):
            p.requires_grad_(False)
        self.static_enc.eval()
        self._aux = None
        self._static_render = None

    def load_pretrained(self, path, dyn_warm_start=True):
        """Load the frozen static encoder + renderer from fl4n.ckpt (or a static-base .pt); optionally warm-start the dynamic tower."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
        enc_key = "static_enc." if any(k.startswith("static_enc.") for k in sd) else "reconstructor."
        enc_sd = {k[len(enc_key):]: v for k, v in sd.items() if k.startswith(enc_key)}
        ren_sd = {k[len("renderer."):]: v for k, v in sd.items() if k.startswith("renderer.")}
        targets = [("static_enc", self.static_enc, enc_sd), ("renderer", self.renderer, ren_sd)]
        if dyn_warm_start:
            targets.append(("dyn_enc", self.dyn_enc, uniform_sample_blocks(enc_sd, len(self.dyn_enc.vggt.aggregator.frame_blocks))))
        for name, mod, s in targets:
            miss, unexp = mod.load_state_dict(s, strict=False)
            print(f"[fl4n:{name}] loaded {path} | missing={len(miss)} unexpected={len(unexp)}")

    def train(self, mode=True):
        super().train(mode)
        self.static_enc.eval()
        return self

    def forward(self, images, rays, cam_token, num_cond_views, tau, dyn_mask=None, return_static_render=False):
        """images/rays/cam_token/tau over [cond K | targets]; returns [cond images | rendered targets]."""
        K = num_cond_views
        cond_imgs, cond_cam, cond_tau = images[:, :K], cam_token[:, :K], tau[:, :K]
        target_rays, target_tau = rays[:, K:], tau[:, K:]
        v_target = target_rays.shape[1]
        self._aux = None
        self._static_render = None

        with torch.no_grad():
            feat_static = self.static_enc(cond_imgs, cond_cam)
        static_kv = einops.rearrange(feat_static, "b v p c -> b (v p) c")
        static_rep = einops.repeat(static_kv, "b n c -> (b v) n c", v=v_target)
        if return_static_render:
            with torch.no_grad():
                self._static_render = self._render(static_rep, target_rays, cond_imgs)

        want_mask = dyn_mask is not None
        out = self.dyn_enc(cond_imgs, cond_cam, static_kv=static_kv, return_tokens=want_mask)
        feat_dyn, time_tok = out[0], out[1]
        if want_mask:
            token_list, patch_start_idx, dpt_imgs = out[2]
            logits = self.mask_head(token_list, dpt_imgs, patch_start_idx).squeeze(2)
            tgt = einops.rearrange(dyn_mask[:, :K], "b v c h w -> (b v) c h w")
            tgt = F.interpolate(tgt, size=logits.shape[-2:], mode="nearest")
            tgt = einops.rearrange(tgt, "(b v) c h w -> b v c h w", b=logits.shape[0]).squeeze(2)
            self._aux = {"dyn_logits": logits, "dyn_target": (tgt > 0.0).float()}

        dyn_rep = einops.repeat(einops.rearrange(feat_dyn, "b v p c -> b (v p) c"), "b n c -> (b v) n c", v=v_target)
        dyn_rep = self.time_module(dyn_rep, interp_time_token(time_tok, cond_tau, target_tau))
        return self._render(static_rep + self.gate(dyn_rep), target_rays, cond_imgs)

    def _render(self, scene_tokens, target_rays, cond_imgs):
        rendered = self.renderer(scene_tokens, target_rays)
        return torch.cat([cond_imgs, rendered.to(cond_imgs.dtype)], dim=1)
