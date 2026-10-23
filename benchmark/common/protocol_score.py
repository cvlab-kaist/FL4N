#!/usr/bin/env python3
"""Unified protocol scorer: PSNR / SSIM / LPIPS-vgg."""
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _to_chw(t01: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(t01)).permute(2, 0, 1).float().clamp(0, 1)


import os
SCORE_OWN_RES = os.environ.get("SCORE_OWN_RES", "0") != "0"


def upsample_to(pred_hwc01: np.ndarray, hw) -> np.ndarray:
    """Bilinear-upsample a [H,W,3] float[0,1] prediction to GT (H,W)."""
    H, W = hw
    if pred_hwc01.shape[:2] == (H, W):
        return pred_hwc01
    x = _to_chw(pred_hwc01)[None]
    y = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    return y[0].permute(1, 2, 0).numpy()


def resize_to(img_hwc01: np.ndarray, hw) -> np.ndarray:
    """Resize a [H,W,3] float[0,1] image to (H,W): area for downscale, bilinear for upscale."""
    H, W = hw
    if img_hwc01.shape[:2] == (H, W):
        return img_hwc01
    x = _to_chw(img_hwc01)[None]
    if H < img_hwc01.shape[0] or W < img_hwc01.shape[1]:
        y = F.interpolate(x, size=(H, W), mode="area")
    else:
        y = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    return y[0].permute(1, 2, 0).numpy()


def resize_mask_to(mask_hw: np.ndarray, hw) -> np.ndarray:
    """Nearest-resize a boolean/float HxW mask to (H,W)."""
    H, W = hw
    if mask_hw.shape[:2] == (H, W):
        return mask_hw
    x = torch.from_numpy(mask_hw.astype(np.float32))[None, None]
    y = F.interpolate(x, size=(H, W), mode="nearest")
    return (y[0, 0].numpy() > 0.5)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def masked_psnr(pred: np.ndarray, gt: np.ndarray, mask_hw: np.ndarray) -> float:
    m = mask_hw.astype(bool)
    if m.sum() == 0:
        return float("nan")
    err = ((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)[m]
    return float(-10.0 * np.log10(max(err.mean(), 1e-12)))


class Scorer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        import lpips
        self.lpips = lpips.LPIPS(net="vgg").to(device).eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        # SSIM: 11x11 gaussian window, data_range 1.0
        from skimage.metrics import structural_similarity
        self._ssim = structural_similarity

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        return float(self._ssim(gt, pred, data_range=1.0, channel_axis=-1,
                                gaussian_weights=True, win_size=11, sigma=1.5,
                                use_sample_covariance=False))

    def lpips_val(self, pred: np.ndarray, gt: np.ndarray) -> float:
        p = _to_chw(pred)[None].to(self.device) * 2 - 1
        g = _to_chw(gt)[None].to(self.device) * 2 - 1
        with torch.no_grad():
            return float(self.lpips(p, g).item())

    def score(self, pred_hwc01: np.ndarray, gt_hwc01: np.ndarray,
              dyn_mask: Optional[np.ndarray] = None, covis_mask: Optional[np.ndarray] = None) -> dict:
        """Score pred vs GT; with SCORE_OWN_RES (default) GT and masks are resized to the pred resolution."""
        gt = np.clip(gt_hwc01, 0, 1).astype(np.float32)
        pred = np.clip(pred_hwc01, 0, 1).astype(np.float32)
        if SCORE_OWN_RES:
            ph, pw = pred.shape[:2]
            if gt.shape[:2] != (ph, pw):
                gt = np.clip(resize_to(gt, (ph, pw)), 0, 1).astype(np.float32)
                if dyn_mask is not None:
                    dyn_mask = resize_mask_to(dyn_mask, (ph, pw))
        else:
            pred = np.clip(upsample_to(pred, gt.shape[:2]), 0, 1).astype(np.float32)
        if covis_mask is not None:
            return self._covis_score(pred, gt, covis_mask)
        out = {"psnr": psnr(pred, gt), "ssim": self.ssim(pred, gt),
               "lpips": self.lpips_val(pred, gt), "gt_hw": list(gt.shape[:2]),
               "score_hw": list(pred.shape[:2])}
        if dyn_mask is not None:
            out["dyn_psnr"] = masked_psnr(pred, gt, dyn_mask)
        return out

    def _covis_score(self, pred, gt, covis_mask):
        """Covisibility-masked PSNR/SSIM/LPIPS at pred res (masked MSE->PSNR; SSIM-map avg over mask; LPIPS on img*mask)."""
        ph, pw = pred.shape[:2]
        m = resize_mask_to(covis_mask.astype(np.float32), (ph, pw)).astype(np.float32)
        ms = float(max(m.sum(), 1.0))
        diff = ((gt - pred) ** 2).mean(2)
        mse = float((diff * m).sum() / ms)
        p = -10.0 * np.log10(max(mse, 1e-10))
        _, smap = self._ssim(gt, pred, win_size=11, gaussian_weights=True, channel_axis=-1,
                             data_range=1.0, full=True)
        s = float((smap.mean(2) * m).sum() / ms)
        lp = self.lpips_val((pred * m[..., None]).astype(np.float32), (gt * m[..., None]).astype(np.float32))
        return {"psnr": p, "ssim": s, "lpips": float(lp), "gt_hw": list(gt.shape[:2]),
                "score_hw": [ph, pw], "covis_frac": float(m.mean())}


def _selftest():
    s = Scorer(device="cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    gt = rng.random((540, 960, 3), dtype=np.float32)
    print("identical (native):", {k: round(v, 3) if isinstance(v, float) else v
                                   for k, v in s.score(gt, gt).items()})
    low = np.asarray(
        torch.nn.functional.interpolate(_to_chw(gt)[None], size=(256, 256),
                                        mode="bilinear", align_corners=False)[0]
        .permute(1, 2, 0))
    print("256->960x540 upsample vs gt:", {k: round(v, 3) if isinstance(v, float) else v
                                            for k, v in s.score(low, gt).items()})
    noisy = np.clip(gt + rng.normal(0, 0.1, gt.shape), 0, 1).astype(np.float32)
    print("gt+noise(0.1):", {k: round(v, 3) if isinstance(v, float) else v
                             for k, v in s.score(noisy, gt).items()})
    m = np.zeros((540, 960), bool); m[100:300, 200:600] = True
    print("with dyn mask:", {k: round(v, 3) if isinstance(v, float) else v
                             for k, v in s.score(noisy, gt, dyn_mask=m).items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
