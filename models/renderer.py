# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the license in the root LICENSE file.
import einops
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.layers.embeddings import PatchEmbed, init_weights_normal
from models.layers.final_layer import FinalLayer
from models.layers.renderer_blocks import BidirectionalCrossAttentionBlock, CrossAttentionBlock


class Renderer(nn.Module):
    """Plucker-ray queries + registers attend to scene tokens through depth-1 bidirectional blocks and one final cross block."""

    def __init__(self, depth, hidden_size, patch_size, num_heads, out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.tgt_embedder = PatchEmbed(patch_size, 6, hidden_size, bias=False)
        self.tgt_norm = nn.LayerNorm(hidden_size, bias=False)
        self.n_registers = 4
        self.per_view_register_tokens = nn.Parameter(torch.zeros(1, self.n_registers, hidden_size, dtype=torch.bfloat16))
        self.renderer_core = BidirectionalCrossAttentionRendererCore(hidden_size, num_heads, depth)
        self.final_layer = FinalLayer(hidden_size=hidden_size, patch_size=patch_size, out_channels=out_channels)
        self.output_act = nn.Sigmoid()
        self.initialize_weights()

    def initialize_weights(self):
        for idx, block in enumerate(self.renderer_core.renderer_blocks):
            block.apply(lambda m, s=0.02 / (2 * (idx + 1)) ** 0.5: init_weights_normal(m, s))
        wc = self.tgt_embedder.proj.weight.data
        nn.init.normal_(wc.view([wc.shape[0], -1]), mean=0.0, std=0.02)
        nn.init.constant_(self.final_layer.linear.weight, 0)

    def forward(self, scene_tokens, target_rays):
        b, v, _, h, w = target_rays.shape
        x = self.tgt_norm(self.tgt_embedder(einops.rearrange(target_rays, "b v c h w -> (b v) c h w")))
        registers = einops.repeat(self.per_view_register_tokens, "n p c -> (n b1) p c", b1=b * v)
        x = self.renderer_core(torch.cat([registers, x], dim=1), scene_tokens)
        x = self.output_act(self.final_layer(x[:, self.n_registers:, :]))
        return einops.rearrange(x, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)", v=v,
                                h=h // self.patch_size, w=w // self.patch_size,
                                p1=self.patch_size, p2=self.patch_size, c=3)


class BidirectionalCrossAttentionRendererCore(nn.Module):
    def __init__(self, hidden_size, num_heads, depth):
        super().__init__()
        self.depth = depth
        self.renderer_blocks = nn.ModuleList(
            [BidirectionalCrossAttentionBlock(hidden_size, num_heads) for _ in range(depth - 1)]
            + [CrossAttentionBlock(hidden_size, num_heads)]
        )

    def _run(self, blk, *args):
        if self.training:
            return checkpoint(blk, *args, use_reentrant=False)
        return blk(*args)

    def forward(self, x, scene_tokens):
        for blk in self.renderer_blocks[:-1]:
            x, scene_tokens = self._run(blk, x, scene_tokens)
        return self._run(self.renderer_blocks[-1], x, scene_tokens)
