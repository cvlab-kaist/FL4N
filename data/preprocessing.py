"""Image loading and crop/resize helpers shared by the datasets."""
import hashlib

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from data import camera_utils


def sampling_rng(split, key):
    """Train: global RNG (fresh draws every epoch); otherwise a RandomState seeded by `key` for deterministic eval."""
    if split == "train":
        return np.random
    return np.random.RandomState(int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16))


def load_images_and_dims(image_paths, frame_indices, to_tensor):
    """Load frames at frame_indices (RGBA -> RGB); returns (list of [3,H,W], (H,W))."""
    images = []
    for fi in frame_indices:
        with open(image_paths[int(fi)], "rb") as f:
            img = Image.open(f)
            img.load()
            t = to_tensor(img)
            images.append(t[:3] if t.shape[0] == 4 else t)
    orig_hw = (images[0].shape[1], images[0].shape[2])
    for image in images:
        assert image.shape[1:] == orig_hw, f"shape mismatch {image.shape}"
    return images, orig_hw


def crop_and_resize(data_arrays, orig_hw, tgt_hw, interpolation=T.InterpolationMode.BILINEAR, clip_bounds=True):
    """Center-crop to the target aspect, resize to tgt_hw; returns [N,C,H,W]."""
    crop = T.CenterCrop(camera_utils.get_full_res_crop_dims_constant_ar(orig_hw, tgt_hw))
    rs = T.Resize(tgt_hw, interpolation=interpolation)
    out = torch.stack([rs(crop(d)) for d in data_arrays])
    return torch.clip(out, 0.0, 1.0) if clip_bounds else out


def compute_tgt_hw(reference_h, aspect_ratio, max_hw):
    """(H, W) from a reference height and W/H aspect, bounded by max_hw, both multiples of 8."""
    tgt_w_ref = reference_h * aspect_ratio
    if tgt_w_ref > max_hw:
        return int(max_hw / aspect_ratio) // 8 * 8, max_hw
    if reference_h > max_hw:
        return max_hw, int(max_hw * aspect_ratio) // 8 * 8
    return int(reference_h) // 8 * 8, int(tgt_w_ref) // 8 * 8


def compute_tgt_hw_keep_aspect(orig_hw, long_side, patch=8):
    """Keep the source aspect: long side -> long_side, both dims floored to multiples of patch."""
    orig_h, orig_w = orig_hw
    if orig_h >= orig_w:
        h, w = long_side, round(long_side * orig_w / orig_h)
    else:
        w, h = long_side, round(long_side * orig_h / orig_w)
    return max(patch, (h // patch) * patch), max(patch, (w // patch) * patch)
