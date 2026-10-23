# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the license in the root LICENSE file.
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from models.layers.attention import Attention


def _attn(hidden_dim, num_heads):
    return Attention(dim=hidden_dim, num_heads=num_heads, qkv_bias=False, fc_bias=False, use_qk_norm=True)


def _mlp(hidden_dim, mlp_ratio):
    return Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * mlp_ratio), bias=False)


class CrossAttentionBlock(nn.Module):
    """Target tokens: self-attention, cross-attention to scene tokens, MLP (CDiT, arxiv 2412.03572)."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, bias=False)
        self.self_attn = _attn(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, bias=False)
        self.norm2_kv = nn.LayerNorm(hidden_dim, bias=False)
        self.cross_attn = _attn(hidden_dim, num_heads)
        self.norm_ffn = nn.LayerNorm(hidden_dim, bias=False)
        self.mlp = _mlp(hidden_dim, mlp_ratio)

    def forward(self, x, scene_tokens):
        x = x + self.self_attn(self.norm1(x), kv=None)
        x = x + self.cross_attn(self.norm2(x), kv=self.norm2_kv(scene_tokens))
        return x + self.mlp(self.norm_ffn(x))


class BidirectionalCrossAttentionBlock(nn.Module):
    """Target <-> scene tokens cross-attention in both directions, each path followed by its own MLP."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1_x = nn.LayerNorm(hidden_dim, bias=False)
        self.self_attn = _attn(hidden_dim, num_heads)
        self.norm2_x = nn.LayerNorm(hidden_dim, bias=False)
        self.cross_attn_x = _attn(hidden_dim, num_heads)
        self.norm3_x = nn.LayerNorm(hidden_dim, bias=False)
        self.mlp_x = _mlp(hidden_dim, mlp_ratio)
        self.norm1_rec = nn.LayerNorm(hidden_dim, bias=False)
        self.cross_attn_rec = _attn(hidden_dim, num_heads)
        self.norm2_rec = nn.LayerNorm(hidden_dim, bias=False)
        self.mlp_rec = _mlp(hidden_dim, mlp_ratio)

    def forward(self, x, scene_tokens):
        x = x + self.self_attn(self.norm1_x(x), kv=None)
        x_norm = self.norm2_x(x)
        rec_norm = self.norm1_rec(scene_tokens)
        x = x + self.cross_attn_x(x_norm, kv=rec_norm)
        scene_tokens = scene_tokens + self.cross_attn_rec(rec_norm, kv=x_norm)
        x = x + self.mlp_x(self.norm3_x(x))
        scene_tokens = scene_tokens + self.mlp_rec(self.norm2_rec(scene_tokens))
        return x, scene_tokens
