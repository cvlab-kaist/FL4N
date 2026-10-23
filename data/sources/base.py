"""Shared base for the (camera x time) grid datasets: sequence bookkeeping, retrying loads, camera conditioning."""
from typing import List, Optional

import numpy as np
import torch
from torchvision import transforms as T

from data import normalization, preprocessing


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, im_size_hw, split="train", num_cond_views=8, num_tgt_views=12, zero_out_cam_cond_p=0.0,
                 root_path="", scenes: Optional[List[str]] = None, train_frac=0.9):
        super().__init__()
        self.im_size_hw = tuple(im_size_hw)
        self.split = split
        self.num_cond_views = num_cond_views
        self.num_tgt_views = num_tgt_views
        self.zero_out_cam_cond_p = zero_out_cam_cond_p
        self.root_path = root_path
        self.scenes_filter = set(scenes) if scenes else None
        self.train_frac = train_frac
        self.to_tensor = T.ToTensor()
        self._seq_meta: dict = {}
        self.sequences: List[str] = []
        self._bad_scenes: set = set()
        self._initialize_sequences()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        for attempt in range(10):
            idx = index if attempt == 0 else np.random.randint(0, len(self))
            seq_name = self.sequences[idx]
            if seq_name in self._bad_scenes:
                continue
            try:
                return self._try_load(seq_name)
            except Exception as e:
                print(f"[{type(self).__name__}] skipping bad scene {seq_name}: {type(e).__name__}: {e}", flush=True)
                self._bad_scenes.add(seq_name)
        raise RuntimeError(f"{type(self).__name__}: no usable scene after 10 retries")

    def _tgt_hw(self):
        return preprocessing.compute_tgt_hw(self.im_size_hw[0], self.im_size_hw[1] / self.im_size_hw[0], max(self.im_size_hw))

    def _zero_cond_pose(self):
        return self.split != "train" or float(np.random.uniform()) <= self.zero_out_cam_cond_p

    def _camera_cond(self, intrinsics, c2w_poses, tgt_hw):
        """Normalize poses and build (plucker rays, pose tokens) for [cond | targets]."""
        c2w_poses, camera_scale = normalization.normalize_extrinsics(c2w_poses, num_cond_views=self.num_cond_views)
        return normalization.build_cam_cond(c2w_poses, intrinsics, num_cond_views=self.num_cond_views, tgt_hw=tgt_hw,
                                            camera_scale=camera_scale, zero_cond_pose=self._zero_cond_pose())

    def _initialize_sequences(self):
        raise NotImplementedError

    def _try_load(self, seq_name):
        raise NotImplementedError
