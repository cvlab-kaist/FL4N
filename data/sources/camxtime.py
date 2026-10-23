"""CamXTime: ~120 static cameras per trajectory, one mp4 (T frames) per camera -> a (camera x time) grid."""
import json
import os
import re
from functools import lru_cache

import av
import numpy as np
import torch
import torch.nn.functional as F

from data import camera_utils, preprocessing
from data.sources.base import SceneDataset

_BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
_CAM_KEY_OFFSET = 32


def _quarter_interp_frames(cond_t):
    """Real frames at 1/4, 1/2, 3/4 of every cond-time gap (novel-time targets); falls back to cond times."""
    ct = sorted(set(int(x) for x in cond_t))
    out = set()
    for a, b in zip(ct, ct[1:]):
        for k in (1, 2, 3):
            f = a + k * (b - a) // 4
            if a < f < b:
                out.add(f)
    return sorted(out) or list(ct)


def _blender_c2w_to_opencv(c2w):
    out = np.array(c2w, dtype=np.float32).copy()
    out[:3, :3] = out[:3, :3] @ _BLENDER_TO_OPENCV
    return out


class CamXTimeDataset(SceneDataset):
    def __init__(self, im_size_hw, split="train", num_cond_views=8, num_tgt_views=12, zero_out_cam_cond_p=0.0,
                 root_path="", num_frames=120, grid_novel_ratio=0.25, grid_novel_time_ratio=0.33,
                 mask_root=None, scenes=None, train_frac=0.9):
        self.num_frames = int(num_frames)
        self.grid_novel_ratio = float(grid_novel_ratio)
        self.grid_novel_time_ratio = float(grid_novel_time_ratio)
        self.mask_root = mask_root
        super().__init__(im_size_hw, split, num_cond_views, num_tgt_views, zero_out_cam_cond_p, root_path, scenes, train_frac)

    @property
    def T(self):
        return self.num_frames

    def _initialize_sequences(self):
        if not os.path.isdir(self.root_path):
            raise RuntimeError(f"CamXTime root missing: {self.root_path}")
        all_seqs = []
        for scene in sorted(os.listdir(self.root_path)):
            scene_dir = os.path.join(self.root_path, scene)
            if not (scene.lower().startswith("scene") and os.path.isdir(scene_dir)):
                continue
            if self.scenes_filter is not None and scene not in self.scenes_filter:
                continue
            for traj in sorted(os.listdir(scene_dir)):
                tdir = os.path.join(scene_dir, traj)
                json_path = os.path.join(tdir, f"{traj}-camera.json")
                if not (traj.startswith("camera-trajectory-") and os.path.isfile(json_path)):
                    continue
                cam_numbers, cam_paths = [], []
                for f in sorted(os.listdir(tdir)):
                    if f.startswith("cam") and f.endswith("_full_motion.mp4"):
                        cam_numbers.append(int(f[3:6]))
                        cam_paths.append(os.path.join(tdir, f))
                if len(cam_numbers) < 2:
                    continue
                seq_name = f"{scene}/{traj}"
                self._seq_meta[seq_name] = {"scene": scene, "json_path": json_path,
                                            "cam_numbers": cam_numbers, "cam_paths": cam_paths}
                all_seqs.append(seq_name)
        if not all_seqs:
            raise RuntimeError(f"No usable CamXTime sequences under {self.root_path}")
        all_seqs.sort()
        scenes = sorted({s.split("/")[0] for s in all_seqs})
        cut = int(len(scenes) * self.train_frac)
        keep = set(scenes[:cut]) if self.split == "train" else set(scenes[cut:])
        self.sequences = [s for s in all_seqs if s.split("/")[0] in keep]
        print(f"[CamXTime] {len(self.sequences)} sequences in split={self.split} (total {len(all_seqs)})")

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        return self._attach_dyn_mask(sample) if self.mask_root is not None else sample

    @lru_cache(maxsize=32)
    def _load_mask_npz(self, seq_name):
        return np.load(os.path.join(self.mask_root, f"{seq_name}.npz"))

    def _attach_dyn_mask(self, sample):
        """Precomputed per-(camera, frame) dynamic masks, stored bit-packed little-endian per real camera."""
        images, cam_enc, ids, cam_token, tau = sample
        V, _, H, W = images.shape
        masks = torch.zeros(V, 1, H, W)
        for v, i in enumerate(ids):
            m = re.search(r"(Scene\d+/camera-trajectory-\d+)_cam(\d+)_t(\d+)", i)
            seq_name, slot, frame = m.group(1), int(m.group(2)), int(m.group(3))
            real_cam = int(self._seq_meta[seq_name]["cam_numbers"][slot])
            packed = self._load_mask_npz(seq_name)[f"mask_cam{real_cam:03d}"][frame]
            um = np.unpackbits(packed, axis=-1, bitorder="little")[:, :W]
            mt = torch.from_numpy(um.astype(np.float32))[None]
            if mt.shape[-2:] != (H, W):
                mt = F.interpolate(mt[None], size=(H, W), mode="nearest")[0]
            masks[v] = mt
        return images, cam_enc, ids, cam_token, tau, masks

    @staticmethod
    def _sample_cam_slots(num_sel, num_cam, rng):
        """Cameras spread over the trajectory: both endpoints plus random interior slots, sorted."""
        if num_sel >= num_cam:
            return list(range(num_cam))
        interior = rng.choice(np.arange(1, num_cam - 1), size=num_sel - 2, replace=False)
        return [0] + sorted(int(x) for x in interior) + [num_cam - 1]

    def _sample_indices_train(self, seq_name):
        """K cond (cam_i, t_i) on a multiple-of-4 time stride; targets = transport / novel-view / novel-time mix."""
        meta = self._seq_meta[seq_name]
        num_cam = len(meta["cam_numbers"])
        rng = preprocessing.sampling_rng(self.split, seq_name)
        K, num_tgt, T = self.num_cond_views, self.num_tgt_views, self.T
        cond_slots = self._sample_cam_slots(K, num_cam, rng)
        stride4 = max(4, ((T - 1) // max(1, K - 1)) // 4 * 4)
        cond_t = [i * stride4 for i in range(K) if i * stride4 <= T - 1]
        K = min(K, len(cond_t))
        cond_slots, cond_t = cond_slots[:K], cond_t[:K]
        ref = cond_t[0]
        c_lo, c_hi = min(cond_slots), max(cond_slots)
        off = [(cond_slots[i], cond_t[j]) for i in range(K) for j in range(K) if i != j]
        novel = [s for s in range(c_lo, c_hi + 1) if s not in set(cond_slots)]
        mids = _quarter_interp_frames(cond_t)
        nT = int(round(num_tgt * self.grid_novel_time_ratio))
        nB = int(round((num_tgt - nT) * self.grid_novel_ratio))
        nA = num_tgt - nT - nB
        tgt = [off[int(rng.randint(len(off)))] for _ in range(nA)]
        for _ in range(nB):
            c = int(rng.choice(novel)) if novel else int(rng.randint(c_lo, c_hi + 1))
            tgt.append((c, int(rng.choice(cond_t))))
        for _ in range(nT):
            tgt.append((int(rng.choice(cond_slots)), int(rng.choice(mids))))
        self.num_cond_views = K
        all_flat = np.array([cond_slots[i] * T + cond_t[i] for i in range(K)] + [c * T + t for c, t in tgt], dtype=int)
        tau = torch.tensor([t - ref for t in cond_t] + [t - ref for _, t in tgt], dtype=torch.float32)
        return meta["cam_paths"], all_flat, tau

    def _sample_indices_eval(self, seq_name):
        """Deterministic N x N grid: cond diagonal, then self, all off-diagonal cells, and midpoint novel times."""
        meta = self._seq_meta[seq_name]
        num_cam = len(meta["cam_numbers"])
        n = min(self.num_cond_views, num_cam, self.T)
        cam_slots = [int(round(k * (num_cam - 1) / max(1, n - 1))) for k in range(n)]
        time_idxs = sorted(set(int(round(k * (self.T - 1) / max(1, n - 1))) for k in range(n)))
        n = min(len(cam_slots), len(time_idxs))
        cam_slots, time_idxs = cam_slots[:n], time_idxs[:n]
        ref = time_idxs[0]
        cond = [(cam_slots[k], time_idxs[k]) for k in range(n)]
        others = [(cam_slots[i], time_idxs[j]) for i in range(n) for j in range(n) if i != j]
        novel = [(cam_slots[k], (time_idxs[k] + time_idxs[k + 1]) // 2) for k in range(n - 1)]
        pairs = cond + cond + others + novel
        self.num_cond_views = n
        all_flat = np.array([c * self.T + t for c, t in pairs], dtype=int)
        tau = torch.tensor([t - ref for _, t in pairs], dtype=torch.float32)
        return meta["cam_paths"], all_flat, tau

    def _sample_indices(self, seq_name):
        if self.split == "train":
            return self._sample_indices_train(seq_name)
        return self._sample_indices_eval(seq_name)

    @staticmethod
    def _decode_frames(path, times):
        want = sorted({int(t) for t in times})
        out = {}
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "NONE"
            for i, frame in enumerate(container.decode(stream)):
                if i in want:
                    out[i] = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).float() / 255.0
                if i >= want[-1]:
                    break
        if not out:
            raise RuntimeError(f"decoded no frames from {path}")
        last = out[max(out)]
        return {t: out.get(t, last) for t in want}

    def _try_load(self, seq_name):
        tgt_hw = self._tgt_hw()
        cam_paths, frame_indices, tau = self._sample_indices(seq_name)
        by_cam = {}
        for fi in frame_indices:
            slot, t = divmod(int(fi), self.T)
            by_cam.setdefault(slot, []).append(t)
        decoded = {slot: self._decode_frames(cam_paths[slot], ts) for slot, ts in by_cam.items()}
        images = [decoded[divmod(int(fi), self.T)[0]][divmod(int(fi), self.T)[1]] for fi in frame_indices]
        orig_hw = (images[0].shape[1], images[0].shape[2])
        images = preprocessing.crop_and_resize(images, orig_hw, tgt_hw)
        intrinsics, c2w_poses = self._load_cameras(seq_name, frame_indices, orig_hw, tgt_hw)
        cam_enc, cam_cond_token = self._camera_cond(intrinsics, c2w_poses, tgt_hw)
        image_ids = [f"{seq_name}_cam{divmod(int(fi), self.T)[0]:03d}_t{divmod(int(fi), self.T)[1]:03d}" for fi in frame_indices]
        return images, cam_enc, image_ids, cam_cond_token, tau

    @lru_cache(maxsize=64)
    def _read_traj(self, json_path):
        with open(json_path) as f:
            return json.load(f)

    def _load_cameras(self, seq_name, frame_indices, im_hw_orig, tgt_hw):
        meta = self._seq_meta[seq_name]
        js = self._read_traj(meta["json_path"])
        crop_hw = camera_utils.get_full_res_crop_dims_constant_ar(im_hw_orig, tgt_hw)
        K = js["intrinsics"]["K"]
        sx, sy = im_hw_orig[1] / float(js["intrinsics"]["width"]), im_hw_orig[0] / float(js["intrinsics"]["height"])
        traj = js["trajectory"]
        traj_keys = sorted(traj, key=int)
        intrinsics, c2w_poses = [], []
        for fi in frame_indices:
            slot, _ = divmod(int(fi), self.T)
            cam_no = meta["cam_numbers"][slot]
            key = str(_CAM_KEY_OFFSET + cam_no)
            if key not in traj:
                key = traj_keys[(cam_no - 1) % len(traj_keys)]
            c2w_poses.append(_blender_c2w_to_opencv(traj[key]["c2w"]))
            fx, fy, cx, cy = camera_utils.adjust_intrinsics_for_crop_and_resize(
                (float(K[0][0]) * sx, float(K[1][1]) * sy, float(K[0][2]) * sx, float(K[1][2]) * sy), im_hw_orig, crop_hw, tgt_hw)
            intrinsics.append([fx, fy, cx, cy])
        return torch.tensor(np.array(intrinsics), dtype=torch.float32), torch.tensor(np.array(c2w_poses), dtype=torch.float32)
