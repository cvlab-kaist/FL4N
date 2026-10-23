"""Lightning data module: primary dataset + resampled secondaries for training, per-dataset validation loaders."""
import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class _SampleAdapter:
    """Normalizes dataset outputs to (images, rays, image_ids, cam_token, tau, dyn_mask, num_cond_views)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        images, rays, image_ids, cam_token, tau = sample[:5]
        dyn_mask = sample[5] if len(sample) > 5 else torch.zeros(images.shape[0], 1, images.shape[-2], images.shape[-1])
        return images, rays, image_ids, cam_token, tau, dyn_mask, self.base.num_cond_views


class _MixedDataset:
    """Primary dataset in full each epoch plus `len(primary) * mix_ratio` random draws from each secondary."""

    def __init__(self, primary, secondaries):
        self.primary = primary
        self.secondaries = [(ds, max(0, round(len(primary) * ratio))) for ds, ratio in secondaries]
        self._breakpoints = np.cumsum([len(primary)] + [n for _, n in self.secondaries]).tolist()
        print(f"[mix] primary={len(primary)} + secondaries={[(len(ds), n) for ds, n in self.secondaries]}")

    def __len__(self):
        return self._breakpoints[-1]

    def __getitem__(self, idx):
        if idx < len(self.primary):
            return self.primary[idx]
        for i, (ds, _) in enumerate(self.secondaries):
            if idx < self._breakpoints[i + 1]:
                return ds[np.random.randint(len(ds))]
        raise IndexError(idx)


def _build_dataset(cfg: DictConfig, split: str):
    ds = hydra.utils.instantiate(cfg.dataset, split=split, im_size_hw=tuple(cfg.im_size_hw),
                                 num_cond_views=cfg.num_cond_views, zero_out_cam_cond_p=cfg.zero_out_cam_cond_p)
    return _SampleAdapter(ds)


def _collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), [b[2] for b in batch],
            torch.stack([b[3] for b in batch]), torch.stack([b[4] for b in batch]), torch.stack([b[5] for b in batch]),
            batch[0][6])


def _short_name(sub_cfg: DictConfig) -> str:
    t = sub_cfg.dataset._target_.rsplit(".", 1)[-1]
    return t[:-7].lower() if t.endswith("Dataset") else t.lower()


class FL4NDataModule(L.LightningDataModule):
    def __init__(self, data_cfg: DictConfig):
        super().__init__()
        self.cfg = data_cfg
        self._train = None
        self._val = []
        self.val_names = []

    def setup(self, stage=None):
        primary = _build_dataset(self.cfg, "train")
        secondaries = [(_build_dataset(sub, "train"), float(sub.mix_ratio)) for sub in (self.cfg.get("secondaries") or [])]
        self._train = _MixedDataset(primary, secondaries) if secondaries else primary
        self._val = [_build_dataset(self.cfg, "test")]
        self.val_names = ["val"]
        for sub in (self.cfg.get("val_only") or []):
            ds = _build_dataset(sub, "test")
            self._val.append(ds)
            self.val_names.append(f"val_{_short_name(sub)}")
            print(f"[val] {self.val_names[-1]}: {len(ds)} samples", flush=True)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle, num_workers=self.cfg.num_workers,
                          pin_memory=self.cfg.pin_memory, drop_last=shuffle, collate_fn=_collate,
                          persistent_workers=self.cfg.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self):
        return [self._loader(ds, shuffle=False) for ds in self._val]
