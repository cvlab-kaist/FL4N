"""NVIDIA Dynamic Scenes (val only): monocular input video, targets = sampled rig cameras x every timestep."""
import os

import numpy as np
import torch
from torchvision import transforms as T

from data import camera_utils, normalization, preprocessing


def _load_calibration(cam_dir):
    ext = np.loadtxt(os.path.join(cam_dir, "extrinsic.txt")).astype(np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = ext[1:4].T
    c2w[:3, 3] = ext[0]
    K = np.loadtxt(os.path.join(cam_dir, "intrinsic.txt")).astype(np.float32)
    return c2w, (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))


class NvidiaDataset(torch.utils.data.Dataset):
    def __init__(self, im_size_hw, split="test", num_cond_views=12, zero_out_cam_cond_p=1.0, root_path="",
                 scenes=None, num_target_cams=3, keep_aspect=True):
        super().__init__()
        self.im_size_hw = tuple(im_size_hw)
        self.max_hw = max(im_size_hw)
        self.split = split
        self.zero_out_cam_cond_p = zero_out_cam_cond_p
        self.root_path = root_path
        self.num_target_cams = int(num_target_cams)
        self.keep_aspect = bool(keep_aspect)
        self.to_tensor = T.ToTensor()
        self._scene_meta = {}
        self._build_index(scenes)
        self.num_cond_views = self._num_input

    def __len__(self):
        return len(self._scenes)

    def _build_index(self, scenes_filter):
        if not os.path.isdir(self.root_path):
            raise RuntimeError(f"NVIDIA root not found: {self.root_path}")
        cands = sorted(d for d in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, d)))
        if scenes_filter is not None:
            cands = [s for s in cands if s in set(scenes_filter)]
        usable, num_input, num_times = [], None, None
        for s in cands:
            sd = os.path.join(self.root_path, s)
            mv, inp, cal, msk = (os.path.join(sd, x) for x in ("multiview_GT", "input_images", "calibration", "foreground_mask"))
            if not all(os.path.isdir(p) for p in (mv, inp, cal, msk)):
                continue
            times = sorted(d for d in os.listdir(mv) if os.path.isdir(os.path.join(mv, d)))
            inputs = sorted(f for f in os.listdir(inp) if f.endswith(".jpg"))
            cams = sorted(d for d in os.listdir(cal) if os.path.isdir(os.path.join(cal, d)))
            if not times or not inputs or not cams:
                continue
            self._scene_meta[s] = {"dir": sd, "times": times, "cams": cams}
            usable.append(s)
            num_input = len(inputs) if num_input is None else num_input
            num_times = len(times) if num_times is None else num_times
        if not usable:
            raise RuntimeError(f"[NVIDIA] no usable scenes under {self.root_path}")
        self._scenes, self._num_input, self._num_times = usable, num_input, num_times
        n_cams = len(self._scene_meta[usable[0]]["cams"])
        self._target_cams = np.linspace(0, n_cams - 1, self.num_target_cams).round().astype(int).tolist()
        print(f"[NVIDIA] scenes={usable} input={num_input} times={num_times} target_cams={self._target_cams}")

    def __getitem__(self, idx):
        scene = self._scenes[idx]
        meta = self._scene_meta[scene]
        sd, times = meta["dir"], meta["times"]
        cond = [(c, c) for c in range(self._num_input)]
        tgt = [(cam, t) for cam in self._target_cams for t in range(self._num_times)]
        views = cond + tgt
        paths = ([os.path.join(sd, "input_images", f"cam{c + 1:02d}.jpg") for (c, _) in cond]
                 + [os.path.join(sd, "multiview_GT", times[t], f"cam{cam + 1:02d}.jpg") for (cam, t) in tgt])
        images, orig_hw = preprocessing.load_images_and_dims(paths, list(range(len(paths))), self.to_tensor)
        if self.keep_aspect:
            tgt_hw = preprocessing.compute_tgt_hw_keep_aspect(orig_hw, long_side=self.max_hw, patch=8)
        else:
            tgt_hw = preprocessing.compute_tgt_hw(self.im_size_hw[0], self.im_size_hw[1] / self.im_size_hw[0], self.max_hw)
        images = preprocessing.crop_and_resize(images, orig_hw, tgt_hw)

        crop_hw = camera_utils.get_full_res_crop_dims_constant_ar(orig_hw, tgt_hw)
        c2ws, intr = [], []
        for (cam, _) in views:
            c2w, fxfycxcy = _load_calibration(os.path.join(sd, "calibration", f"cam{cam + 1:02d}"))
            c2ws.append(c2w)
            intr.append(list(camera_utils.adjust_intrinsics_for_crop_and_resize(fxfycxcy, orig_hw, crop_hw, tgt_hw)))
        c2ws = torch.tensor(np.array(c2ws), dtype=torch.float32)
        intr = torch.tensor(np.array(intr), dtype=torch.float32)
        c2ws, camera_scale = normalization.normalize_extrinsics(c2ws, num_cond_views=self.num_cond_views)
        zero = self.split != "train" or float(np.random.uniform()) <= self.zero_out_cam_cond_p
        cam_enc, cam_cond_token = normalization.build_cam_cond(c2ws, intr, num_cond_views=self.num_cond_views, tgt_hw=tgt_hw,
                                                               camera_scale=camera_scale, zero_cond_pose=zero)
        t0 = cond[0][1]
        tau = torch.tensor([t - t0 for (_, t) in views], dtype=torch.float32)

        dyn = torch.zeros(len(views), 1, tgt_hw[0], tgt_hw[1], dtype=torch.float32)
        mask_paths = [os.path.join(sd, "foreground_mask", f"cam{c + 1:02d}.png") for (c, _) in cond]
        mviews, _ = preprocessing.load_images_and_dims(mask_paths, list(range(len(mask_paths))), self.to_tensor)
        mviews = preprocessing.crop_and_resize([m[:1] for m in mviews], orig_hw, tgt_hw, interpolation=T.InterpolationMode.NEAREST)
        dyn[: self.num_cond_views] = (mviews > 0.5).float()

        image_ids = [f"nvidia_{scene}_cam{c + 1:02d}_t{t:02d}" for (c, t) in views]
        return images, cam_enc, image_ids, cam_cond_token, tau, dyn
