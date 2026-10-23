"""Rendering loss: target L2 + VGG perceptual, dynamic-region perceptual/gradient terms, mask-head BCE."""
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PerceptualLoss(nn.Module):
    IMAGENET_MEAN = (123.6800 / 255, 116.7790 / 255, 103.9390 / 255)
    FEATURE_SCALE_DIVISORS = (2.6, 4.8, 3.7, 5.6, 1.5 / 10)
    FEATURE_LAYER_INDICES = (3, 8, 13, 22, 31)

    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg19(weights="IMAGENET1K_V1")
        for layer in self.vgg.features[max(self.FEATURE_LAYER_INDICES) + 1:]:
            del layer
        self.register_buffer("_normalization_mean", torch.FloatTensor(self.IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer("_feature_scales", torch.FloatTensor(self.FEATURE_SCALE_DIVISORS))

    def _features(self, x):
        feats = []
        for i, layer in enumerate(self.vgg.features):
            x = layer(x)
            if i in self.FEATURE_LAYER_INDICES:
                feats.append(x)
        return feats

    def forward(self, pred, target):
        """Per-image perceptual distance [N] between [0,1] RGB batches."""
        pred = (pred - self._normalization_mean) * 255.0
        target = (target - self._normalization_mean) * 255.0
        l1 = lambda a, b: torch.mean(torch.abs(a - b), dim=[1, 2, 3])
        feat_err = [l1(ft, fp) / self._feature_scales[i] for i, (ft, fp) in enumerate(zip(self._features(target), self._features(pred)))]
        return (l1(target, pred) + sum(feat_err)) / 255.0


def _dyn_crops(pred, target, mask):
    """Yield (pred_crop, gt_crop) [3,h,w] over the dynamic-mask bounding box of each target view."""
    for bi in range(pred.shape[0]):
        for vi in range(pred.shape[1]):
            m = mask[bi, vi, 0] > 0.5
            if int(m.sum()) < 4:
                continue
            ys, xs = torch.where(m)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            yield pred[bi, vi, :, y0:y1, x0:x1], target[bi, vi, :, y0:y1, x0:x1]


class RenderingLoss(nn.Module):
    def __init__(self, l2_weight=1.0, perceptual_weight=0.5, dyn_perceptual_weight=2.0,
                 dyn_perceptual_size=128, dyn_grad_weight=1.0, dyn_mask_weight=1.0):
        super().__init__()
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        self.dyn_perceptual_weight = dyn_perceptual_weight
        self.dyn_perceptual_size = dyn_perceptual_size
        self.dyn_grad_weight = dyn_grad_weight
        self.dyn_mask_weight = dyn_mask_weight
        self.perceptual = PerceptualLoss().eval()
        for p in self.perceptual.parameters():
            p.requires_grad_(False)

    def forward(self, output, images, num_cond_views, dyn_mask=None):
        """output/images [b,V,3,h,w] over [cond | targets]; dyn_mask [b,V,1,h,w] or None. Returns a dict of terms."""
        pred, target = output[:, num_cond_views:], images[:, num_cond_views:]
        pred_flat = einops.rearrange(pred, "b v c h w -> (b v) c h w")
        target_flat = einops.rearrange(target, "b v c h w -> (b v) c h w")
        l2 = F.mse_loss(pred_flat, target_flat)
        perc = self.perceptual(pred_flat, target_flat).mean()
        total = self.l2_weight * l2 + self.perceptual_weight * perc
        zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
        dyn_perc, dyn_grad = zero, zero
        if dyn_mask is not None:
            mask = dyn_mask[:, num_cond_views:]
            dyn_perc = self._dyn_perceptual(pred, target, mask)
            dyn_grad = self._dyn_grad(pred, target, mask)
            total = total + self.dyn_perceptual_weight * dyn_perc + self.dyn_grad_weight * dyn_grad
        return {"loss": total, "l2": l2, "psnr": -10.0 * torch.log10(l2.detach().clamp(min=1e-10)),
                "perceptual": perc, "dyn_perceptual": dyn_perc, "dyn_grad": dyn_grad}

    def _dyn_perceptual(self, pred, target, mask):
        s = self.dyn_perceptual_size
        crops = [(F.interpolate(p[None], size=(s, s), mode="bilinear", align_corners=False),
                  F.interpolate(t[None], size=(s, s), mode="bilinear", align_corners=False))
                 for p, t in _dyn_crops(pred, target, mask)]
        if not crops:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        P = torch.cat([c[0] for c in crops], 0).clamp(0, 1)
        T = torch.cat([c[1] for c in crops], 0).clamp(0, 1)
        return self.perceptual(P, T).mean()

    def _dyn_grad(self, pred, target, mask):
        losses = []
        for cp, ct in _dyn_crops(pred, target, mask):
            if cp.shape[-1] < 2 or cp.shape[-2] < 2:
                continue
            gx = (cp[:, :, 1:] - cp[:, :, :-1] - (ct[:, :, 1:] - ct[:, :, :-1])).abs().mean()
            gy = (cp[:, 1:, :] - cp[:, :-1, :] - (ct[:, 1:, :] - ct[:, :-1, :])).abs().mean()
            losses.append(gx + gy)
        if not losses:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return torch.stack(losses).mean()


def mask_head_loss(logits, target):
    """Mean BCE over samples that have a non-empty dynamic mask, plus their mean IoU at 0.5; (None, None) if none."""
    logits, target = logits.float(), target.float()
    valid = (target.flatten(1).sum(1) > 0).float()
    if float(valid.sum()) == 0:
        return None, None
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").flatten(1).mean(1)
    with torch.no_grad():
        pb = (torch.sigmoid(logits) > 0.5).flatten(1).float()
        tb = (target > 0.5).flatten(1).float()
        inter = (pb * tb).sum(1)
        iou = inter / (pb.sum(1) + tb.sum(1) - inter).clamp(min=1.0)
    return (bce * valid).sum() / valid.sum(), (iou * valid).sum() / valid.sum()
