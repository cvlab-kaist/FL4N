# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the VGGT license.
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from vggt.layers.block import Block
from vggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from vggt.layers.vision_transformer import vit_large

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class CrossAttention(nn.Module):
    """Query tokens attend to static key/value tokens; zero-init output projection (no-op at step 0)."""

    def __init__(self, dim, kv_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(kv_dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, kv):
        B, Nq, C = x.shape
        Nkv = kv.shape[1]
        q = self.to_q(self.norm_q(x)).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = (self.to_kv(self.norm_kv(kv))
                .reshape(B, Nkv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0))
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Nq, C)
        return self.proj(out)


class Aggregator(nn.Module):
    """VGGT alternating frame/global attention over DINOv2 patch tokens, with optional static cross-attention and time token."""

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
                 num_register_tokens=4, qkv_bias=True, proj_bias=True, ffn_bias=True, qk_norm=True,
                 rope_freq=100, init_values=0.01, cross_attn_static=False, cross_attn_kv_dim=768, time_token=False):
        super().__init__()
        self.time_token = time_token
        self.global_cross = (
            nn.ModuleList([CrossAttention(embed_dim, cross_attn_kv_dim, num_heads) for _ in range(depth)])
            if cross_attn_static else None
        )
        self.patch_embed = vit_large(
            img_size=img_size, patch_size=patch_size, num_register_tokens=num_register_tokens,
            interpolate_antialias=True, interpolate_offset=0.0, block_chunks=0, init_values=1.0,
        )
        if hasattr(self.patch_embed, "mask_token"):
            self.patch_embed.mask_token.requires_grad_(False)

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        def block():
            return Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                         proj_bias=proj_bias, ffn_bias=ffn_bias, init_values=init_values, qk_norm=qk_norm, rope=self.rope)

        self.frame_blocks = nn.ModuleList([block() for _ in range(depth)])
        self.global_blocks = nn.ModuleList([block() for _ in range(depth)])
        self.depth = depth
        self.patch_size = patch_size

        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        if time_token:
            self.time_conditioning_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.time_token_idx = 1
        self.patch_start_idx = 1 + num_register_tokens + (1 if time_token else 0)
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    def forward(self, images, camera_cond=None, static_kv=None):
        B, S, C_in, H, W = images.shape
        images = ((images - self._resnet_mean) / self._resnet_std).view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        _, P, C = patch_tokens.shape

        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        if camera_cond is not None:
            camera_token = camera_token + einops.rearrange(camera_cond, "b s np c -> (b s) np c")
        specials = [camera_token]
        if self.time_token:
            specials.append(self.time_conditioning_token.expand(B * S, 1, C).to(patch_tokens.dtype))
        specials.append(register_token)
        tokens = torch.cat(specials + [patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device) + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        _, P, C = tokens.shape

        output_list = []
        for i in range(self.depth):
            tokens, frame_inter = self._frame_attention(tokens, B, S, P, C, i, pos)
            tokens, global_inter = self._global_attention(tokens, B, S, P, C, i, pos, static_kv)
            output_list.append(torch.cat([frame_inter, global_inter], dim=-1))
        return output_list, self.patch_start_idx

    def _run(self, blk, tokens, pos):
        if self.training:
            return checkpoint(blk, tokens, pos, use_reentrant=False)
        return blk(tokens, pos=pos)

    def _frame_attention(self, tokens, B, S, P, C, idx, pos):
        tokens = tokens.view(B * S, P, C)
        pos = pos.view(B * S, P, 2) if pos is not None else None
        tokens = self._run(self.frame_blocks[idx], tokens, pos)
        return tokens, tokens.view(B, S, P, C)

    def _global_attention(self, tokens, B, S, P, C, idx, pos, static_kv):
        tokens = tokens.view(B, S * P, C)
        pos = pos.view(B, S * P, 2) if pos is not None else None
        tokens = self._run(self.global_blocks[idx], tokens, pos)
        if self.global_cross is not None and static_kv is not None:
            tokens = tokens + self.global_cross[idx](tokens, static_kv)
        return tokens, tokens.view(B, S, P, C)


def slice_expand_and_flatten(token_tensor, B, S):
    """(1, 2, X, C) special tokens -> (B*S, X, C): slot 0 for the first frame, slot 1 for the rest."""
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    return combined.view(B * S, *combined.shape[2:])
