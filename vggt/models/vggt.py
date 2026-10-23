# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the VGGT license.
import torch
import torch.nn as nn

from vggt.models.aggregator import Aggregator


class VGGT(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24,
                 cross_attn_static=False, cross_attn_kv_dim=768, time_token=False):
        super().__init__()
        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth,
            cross_attn_static=cross_attn_static, cross_attn_kv_dim=cross_attn_kv_dim, time_token=time_token,
        )

    def forward(self, images, camera_cond=None, static_kv=None, return_time_token=False, return_tokens=False):
        if images.dim() == 4:
            images = images.unsqueeze(0)
        token_list, patch_start_idx = self.aggregator(images, camera_cond, static_kv=static_kv)
        out = [token_list[-1]]
        if return_time_token:
            out.append(token_list[-1][:, :, self.aggregator.time_token_idx])
        if return_tokens:
            out += [token_list, patch_start_idx]
        return out[0] if len(out) == 1 else tuple(out)
