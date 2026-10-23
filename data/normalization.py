"""Scene normalization (recenter on the first camera, camera-baseline scale) and camera conditioning tokens."""
import os

import torch

from data import camera_utils
from vggt.utils import pose_enc

CAMERA_SCALE_MULTIPLIER = float(os.environ.get("CAM_SCALE", 1.35))


def normalize_extrinsics(c2ws, num_cond_views):
    """Express all c2w in the first camera's frame and scale so the max cond-camera distance is CAMERA_SCALE_MULTIPLIER."""
    c2ws = torch.linalg.inv(c2ws[0].clone())[None, ...] @ c2ws
    w1_raw = float(torch.max(torch.norm(c2ws[:num_cond_views, :3, 3], dim=-1))) if num_cond_views > 1 else 0.0
    lam = CAMERA_SCALE_MULTIPLIER / w1_raw if w1_raw > 1e-6 else 1.0
    c2ws[:, :3, 3] *= lam
    return c2ws, torch.max(torch.norm(c2ws[:num_cond_views, :3, 3], dim=-1))


def build_cam_cond(c2w_poses, intrinsics_fxfycxcy_px, num_cond_views, tgt_hw, camera_scale, zero_cond_pose):
    """Plucker rays [V,6,H,W] and pose tokens [V,11]; cond rays/tokens are zeroed for pose-free conditioning."""
    cam_cond_token = pose_enc.extri_intri_to_pose_encoding(
        c2w_poses.unsqueeze(0), intrinsics_fxfycxcy_px.unsqueeze(0), image_size_hw=tgt_hw).squeeze(0)
    Ks = camera_utils.get_K_matrices(intrinsics_fxfycxcy_px)
    cam_enc = camera_utils.compute_plucker_rays(c2w_poses, Ks, tgt_hw)
    if zero_cond_pose:
        cam_enc[:num_cond_views] *= 0.0
        cam_cond_token[:num_cond_views] *= 0.0
    scale_tokens = torch.tensor([float(camera_scale), 0.0]).unsqueeze(0).expand(cam_cond_token.shape[0], -1)
    return cam_enc, torch.cat([cam_cond_token, scale_tokens], dim=-1)
