"""AdaLN transformer on the dynamic latent, conditioned on the (interpolated) encoder time token."""
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from torch.utils.checkpoint import checkpoint

from models.layers.attention import Attention
from models.layers.time_modulation import TimeModulation, modulate


class _AdaLNBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, bias=False)
        self.attn = Attention(dim=hidden_dim, num_heads=num_heads, qkv_bias=False, use_qk_norm=True)
        self.norm2 = nn.LayerNorm(hidden_dim, bias=False)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * mlp_ratio), bias=False)
        self.modulation = TimeModulation(hidden_dim, hidden_dim, n_sublayers=2)

    def forward(self, x, cond):
        m = self.modulation(cond)
        x = x + m[:, 0, 2].unsqueeze(1) * self.attn(modulate(self.norm1(x), m[:, 0, 0], m[:, 0, 1]), kv=None)
        return x + m[:, 1, 2].unsqueeze(1) * self.mlp(modulate(self.norm2(x), m[:, 1, 0], m[:, 1, 1]))


class TimeModule(nn.Module):
    def __init__(self, hidden_dim, num_heads, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([_AdaLNBlock(hidden_dim, num_heads) for _ in range(depth)])

    def forward(self, x, cond):
        for blk in self.blocks:
            x = checkpoint(blk, x, cond, use_reentrant=False) if self.training else blk(x, cond)
        return x
