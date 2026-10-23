"""LR schedules (warmup + cosine / step / constant) and the AdamW weight-decay split."""
from functools import partial

import numpy as np
import torch


def split_decay_params(model: torch.nn.Module):
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay if p.dim() == 1 else decay).append(p)
    return decay, nodecay


def warmup_constant_lambda(iter_idx, warmup_steps):
    return float(iter_idx) / float(max(1, warmup_steps)) if iter_idx < warmup_steps else 1.0


def warmup_cosine_lambda(iter_idx, warmup_steps, hold_steps, num_iter_total, cosine_min_range=0.0):
    if iter_idx < warmup_steps:
        return float(iter_idx) / float(max(1, warmup_steps))
    if iter_idx <= warmup_steps + hold_steps:
        return 1.0
    decay_iters = num_iter_total - warmup_steps - hold_steps
    cosine = 0.5 * np.cos(np.pi * (iter_idx - warmup_steps - hold_steps) / max(1, decay_iters)) + 0.5
    return cosine_min_range + cosine * (1.0 - cosine_min_range)


def warmup_step_lambda(iter_idx, warmup_steps, hold_steps, step_factor=0.1):
    if iter_idx < warmup_steps:
        return float(iter_idx) / float(max(1, warmup_steps))
    return 1.0 if iter_idx <= warmup_steps + hold_steps else step_factor


def get_lr_lambda(name, max_steps, sch_cfg):
    if name == "constant":
        return partial(warmup_constant_lambda, warmup_steps=sch_cfg.warmup_steps)
    if name == "cosine":
        return partial(warmup_cosine_lambda, warmup_steps=sch_cfg.warmup_steps, hold_steps=sch_cfg.hold_steps,
                       num_iter_total=max_steps, cosine_min_range=sch_cfg.cosine_min_range)
    if name == "step":
        return partial(warmup_step_lambda, warmup_steps=sch_cfg.warmup_steps, hold_steps=sch_cfg.hold_steps,
                       step_factor=sch_cfg.step_factor)
    raise ValueError(f"Unknown scheduler {name!r}")
