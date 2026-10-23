"""Kubric4D: per scene ~16 moving-camera views x 60 frames, per-view JSON with per-frame poses and instance motion."""
import json
import os
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from data import camera_utils, preprocessing
from data.sources.base import SceneDataset

_BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def _quat_wxyz_to_R(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _blender_c2w_to_opencv(pos, quat_wxyz):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _quat_wxyz_to_R(quat_wxyz) @ _BLENDER_TO_OPENCV
    c2w[:3, 3] = np.asarray(pos, dtype=np.float32)
    return c2w


def _normalized_K_to_px(K_norm, im_w, im_h):
    return (float(K_norm[0][0]) * im_w, abs(float(K_norm[1][1])) * im_h,
            -float(K_norm[0][2]) * im_w, -float(K_norm[1][2]) * im_h)


def _pair_angle_deg(q0, q1):
    q0 = q0 / (np.linalg.norm(q0) + 1e-8)
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(float(q0 @ q1)), 0.0, 1.0))))


class Kubric4DDataset(SceneDataset):
    def __init__(self, im_size_hw, split="train", num_cond_views=8, num_tgt_views=12, zero_out_cam_cond_p=0.0,
                 root_path="", split_subdir="train", grid_novel_ratio=0.25, grid_novel_time_ratio=0.33,
                 stride_choices=(4, 8), mask_roots=None, return_dyn_mask=True, vel_thresh=0.03,
                 rot_step_thresh=6.0, eval_cond_pairs=None, eval_target_pairs=None, scenes=None, train_frac=0.9):
        self.split_subdir = split_subdir
        self.grid_novel_ratio = float(grid_novel_ratio)
        self.grid_novel_time_ratio = float(grid_novel_time_ratio)
        self.stride_choices = list(stride_choices)
        self.mask_roots = {int(k): v for k, v in mask_roots.items()} if mask_roots else None
        self._mask_stride = None
        self.return_dyn_mask = return_dyn_mask
        self.vel_thresh = vel_thresh
        self.rot_step_thresh = rot_step_thresh
        self.eval_cond_pairs = [list(p) for p in eval_cond_pairs] if eval_cond_pairs else None
        self.eval_target_pairs = [list(p) for p in eval_target_pairs] if eval_target_pairs else None
        super().__init__(im_size_hw, split, num_cond_views, num_tgt_views, zero_out_cam_cond_p, root_path, scenes, train_frac)

    def _metadata_path(self, scene, view_dir):
        return os.path.join(self.root_path, self.split_subdir, scene, f"{scene}_{view_dir[len('frames_'):]}.json")

    def _initialize_sequences(self):
        scenes_root = os.path.join(self.root_path, self.split_subdir)
        if not os.path.isdir(scenes_root):
            raise RuntimeError(f"Kubric4D root missing '{self.split_subdir}/' subdir: {self.root_path}")
        all_seqs = []
        for scene in sorted(os.listdir(scenes_root)):
            sdir = os.path.join(scenes_root, scene)
            if not os.path.isdir(sdir) or (self.scenes_filter is not None and scene not in self.scenes_filter):
                continue
            view_dirs = sorted((d for d in os.listdir(sdir) if d.startswith("frames_") and "_v" in d and os.path.isdir(os.path.join(sdir, d))),
                               key=lambda d: int(d.rsplit("_v", 1)[1]))
            T_min, ok = None, bool(view_dirs)
            for vd in view_dirs:
                if not os.path.isfile(self._metadata_path(scene, vd)):
                    ok = False
                    break
                n = len([f for f in os.listdir(os.path.join(sdir, vd)) if f.startswith("rgba_") and f.endswith(".png")])
                T_min = n if T_min is None else min(T_min, n)
            if not ok or T_min is None or T_min < 2:
                continue
            flat_paths = [os.path.join(sdir, vd, f"rgba_{frame:05d}.png") for vd in view_dirs for frame in range(T_min)]
            self._seq_meta[scene] = {"scene": scene, "view_dirs": view_dirs, "T": T_min, "flat_paths": flat_paths}
            all_seqs.append(scene)
        if not all_seqs:
            raise RuntimeError(f"No usable Kubric4D scenes under {scenes_root}")
        all_seqs.sort()
        cutoff = int(len(all_seqs) * self.train_frac)
        if self.split_subdir != "train":
            self.sequences = all_seqs
        else:
            self.sequences = all_seqs[:cutoff] if self.split == "train" else all_seqs[cutoff:]
        print(f"[Kubric4D] {len(self.sequences)} scenes in split={self.split} subdir={self.split_subdir} (usable total {len(all_seqs)})")

    def _sample_indices(self, seq_name):
        if self.split != "train" and self.eval_cond_pairs and self.eval_target_pairs:
            return self._sample_indices_explicit(seq_name)
        return self._sample_indices_train(seq_name)

    def _sample_indices_explicit(self, seq_name):
        meta = self._seq_meta[seq_name]
        T_total, nv = meta["T"], len(meta["view_dirs"])
        cond = [(v, t) for v, t in self.eval_cond_pairs if v < nv and t < T_total]
        tgt = [(v, t) for v, t in self.eval_target_pairs if v < nv and t < T_total]
        if not cond or not tgt:
            raise RuntimeError(f"no valid eval pairs for {seq_name}")
        pairs = cond + tgt
        self.num_cond_views = len(cond)
        all_flat = np.array([v * T_total + t for v, t in pairs], dtype=int)
        tau = torch.tensor([t - cond[0][1] for _, t in pairs], dtype=torch.float32)
        return meta["flat_paths"], all_flat, tau

    def _sample_indices_train(self, seq_name):
        """K random rig views at a strided time window; targets = transport / novel-view / quarter-gap novel-time mix."""
        meta = self._seq_meta[seq_name]
        T, num_v, flat_paths = meta["T"], len(meta["view_dirs"]), meta["flat_paths"]
        rng = preprocessing.sampling_rng(self.split, seq_name)
        K = min(self.num_cond_views, num_v)
        stride = int(rng.choice(self.stride_choices))
        self._mask_stride = stride
        span = stride * (K - 1)
        if T < span + 1:
            stride = max(1, (T - 1) // max(1, K - 1))
            span = stride * (K - 1)
        t_0 = int(rng.randint(0, T - span)) if T - span > 0 else 0
        cond_t = [t_0 + i * stride for i in range(K)]
        cond_v = sorted(int(x) for x in rng.choice(np.arange(num_v), size=K, replace=False))
        num_tgt = self.num_tgt_views
        nT = int(round(num_tgt * self.grid_novel_time_ratio))
        n_novel = int(round((num_tgt - nT) * self.grid_novel_ratio))
        n_trans = num_tgt - nT - n_novel
        off = [(cond_v[i], cond_t[j]) for i in range(K) for j in range(K) if i != j]
        v_lo, v_hi = min(cond_v), max(cond_v)
        novel = [v for v in range(v_lo, v_hi + 1) if v not in set(cond_v)]
        mids = [cond_t[i] + k * stride // 4 for i in range(K - 1) for k in (1, 2, 3) if 0 < k * stride // 4 < stride]
        tgt = [off[int(rng.randint(len(off)))] for _ in range(n_trans)]
        for _ in range(n_novel):
            v = int(rng.choice(novel)) if novel else int(rng.randint(v_lo, v_hi + 1))
            tgt.append((v, int(rng.choice(cond_t))))
        for _ in range(nT):
            t = int(rng.choice(mids)) if mids else int(rng.choice(cond_t))
            tgt.append((int(rng.choice(cond_v)), t))
        self.num_cond_views = K
        all_flat = np.array([cond_v[i] * T + cond_t[i] for i in range(K)] + [v * T + t for v, t in tgt], dtype=int)
        tau = torch.tensor([t - cond_t[0] for t in cond_t] + [t - cond_t[0] for _, t in tgt], dtype=torch.float32)
        return flat_paths, all_flat, tau

    def _try_load(self, seq_name):
        tgt_hw = self._tgt_hw()
        flat_paths, frame_indices, tau = self._sample_indices(seq_name)
        images, orig_hw = preprocessing.load_images_and_dims(flat_paths, frame_indices, self.to_tensor)
        images = preprocessing.crop_and_resize(images, orig_hw, tgt_hw)
        intrinsics, c2w_poses = self._load_cameras(seq_name, frame_indices, orig_hw, tgt_hw)
        cam_enc, cam_cond_token = self._camera_cond(intrinsics, c2w_poses, tgt_hw)
        image_ids = [f"{seq_name}_{os.path.basename(os.path.dirname(flat_paths[int(fi)]))}"
                     f"_{os.path.basename(flat_paths[int(fi)]).rsplit('.', 1)[0]}" for fi in frame_indices]
        if self.return_dyn_mask:
            dyn_mask = self._load_dyn_masks(seq_name, frame_indices, flat_paths, orig_hw, tgt_hw)
            return images, cam_enc, image_ids, cam_cond_token, tau, dyn_mask
        return images, cam_enc, image_ids, cam_cond_token, tau

    def _load_dyn_masks(self, seq_name, frame_indices, flat_paths, orig_hw, tgt_hw):
        """Flow-based masks matching the sampled stride when mask_roots is set, else moving-instance segmentation masks."""
        masks = []
        if self.mask_roots:
            stride = self._mask_stride if self._mask_stride in self.mask_roots else max(self.mask_roots)
            z = np.load(os.path.join(self.mask_roots[stride], f"{seq_name}.npz"))
            T_total, Wn = self._seq_meta[seq_name]["T"], int(z["mask_shape_thw"][2])
            for fi in frame_indices:
                v, f = divmod(int(fi), T_total)
                um = np.unpackbits(z[f"mask_v{v}"][f], axis=-1, count=Wn, bitorder="little")
                masks.append(torch.from_numpy(um.astype(np.float32))[None])
        else:
            for fi, ids in zip(frame_indices, self._per_frame_dyn_ids(seq_name, frame_indices)):
                rgba_path = flat_paths[int(fi)]
                seg_path = os.path.join(os.path.dirname(rgba_path), os.path.basename(rgba_path).replace("rgba_", "segmentation_"))
                with open(seg_path, "rb") as f:
                    seg = np.array(Image.open(f))
                if seg.ndim == 3:
                    seg = seg[..., 0]
                m = np.isin(seg, list(ids)).astype(np.float32) if ids else np.zeros(seg.shape, np.float32)
                masks.append(torch.from_numpy(m)[None])
        return preprocessing.crop_and_resize(masks, orig_hw, tgt_hw, interpolation=T.InterpolationMode.NEAREST, clip_bounds=False)

    def _per_frame_dyn_ids(self, seq_name, frame_indices):
        """Segmentation values (instance idx + 1) of instances moving between consecutive fetched frames."""
        meta = self._seq_meta[seq_name]
        T_total, view_dirs, scene = meta["T"], meta["view_dirs"], meta["scene"]
        clip_t = [int(fi) % T_total for fi in frame_indices]
        view_i = [int(fi) // T_total for fi in frame_indices]
        out = []
        for k, t_cur in enumerate(clip_t):
            t_prev = clip_t[k - 1] if k > 0 else (clip_t[1] if len(clip_t) > 1 else clip_t[0])
            gap = max(1, abs(t_cur - t_prev))
            md = self._read_metadata(self._metadata_path(scene, view_dirs[view_i[k]]))
            ids = set()
            for i, o in enumerate(md.get("instances", [])):
                p = np.asarray(o.get("positions", []), dtype=np.float32)
                if p.shape[0] <= max(t_cur, t_prev):
                    continue
                trans = float(np.linalg.norm(p[t_cur] - p[t_prev])) / gap
                q = np.asarray(o.get("quaternions", []), dtype=np.float32)
                rot = _pair_angle_deg(q[t_cur], q[t_prev]) / gap if q.shape[0] > max(t_cur, t_prev) else 0.0
                if trans > self.vel_thresh or rot > self.rot_step_thresh:
                    ids.add(i + 1)
            out.append(ids)
        return out

    @lru_cache(maxsize=512)
    def _read_metadata(self, meta_path):
        with open(meta_path) as f:
            return json.load(f)

    def _load_cameras(self, seq_name, frame_indices, im_hw_orig, tgt_hw):
        meta = self._seq_meta[seq_name]
        T_total, view_dirs, scene = meta["T"], meta["view_dirs"], meta["scene"]
        crop_hw = camera_utils.get_full_res_crop_dims_constant_ar(im_hw_orig, tgt_hw)
        intrinsics, c2w_poses = [], []
        for flat_idx in frame_indices:
            vi, frame = divmod(int(flat_idx), T_total)
            cam = self._read_metadata(self._metadata_path(scene, view_dirs[vi]))["camera"]
            c2w_poses.append(_blender_c2w_to_opencv(cam["positions"][frame], cam["quaternions"][frame]))
            fx, fy, cx, cy = camera_utils.adjust_intrinsics_for_crop_and_resize(
                _normalized_K_to_px(cam["K"], im_hw_orig[1], im_hw_orig[0]), im_hw_orig, crop_hw, tgt_hw)
            intrinsics.append([fx, fy, cx, cy])
        return torch.tensor(np.array(intrinsics), dtype=torch.float32), torch.tensor(np.array(c2w_poses), dtype=torch.float32)
