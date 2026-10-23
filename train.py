"""Train / validate FL4N (Hydra + Lightning)."""
import datetime
import json
import math
import os
import sys
import warnings
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from data.data_module import FL4NDataModule
from eval.metrics import MetricsComputer
from eval.vis import make_compare_grid, make_grid_vis, make_nvidia_grid
from loss import mask_head_loss
from models.checkpoint import remap_keys
from train_utils import get_lr_lambda, split_decay_params

warnings.filterwarnings("ignore", message="No device id is provided", module="torch.distributed")
torch.set_float32_matmul_precision(precision="high")
torch.multiprocessing.set_sharing_strategy("file_system")
_orig_torch_load = torch.load


def _torch_load_full(*a, **k):
    k["weights_only"] = False
    return _orig_torch_load(*a, **k)


torch.load = _torch_load_full


class FL4NModule(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})
        self.cfg = cfg
        self.model = hydra.utils.instantiate(cfg.model)
        self.loss_fn = hydra.utils.instantiate(cfg.loss)
        self._metrics = None
        self.automatic_optimization = False
        self.strict_loading = False

    def on_save_checkpoint(self, checkpoint):
        """Frozen static encoder + renderer are dropped; they reload from model.pretrained_path (fl4n.ckpt) at init."""
        sd = checkpoint.get("state_dict", {})
        for k in [k for k in sd if k.startswith("model.static_enc.") or k.startswith("model.renderer.")]:
            del sd[k]

    def on_load_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = remap_keys(checkpoint["state_dict"])

    def training_step(self, batch, batch_idx):
        images, rays, image_ids, cam_token, tau, dyn_mask, K = batch
        K = int(K)
        vis_every = int(self.cfg.get("vis_every_n_steps", 500))
        is_vis = vis_every > 0 and self.global_step % vis_every == 0

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(images, rays, cam_token, num_cond_views=K, tau=tau, dyn_mask=dyn_mask, return_static_render=is_vis)
            terms = self.loss_fn(output, images, num_cond_views=K, dyn_mask=dyn_mask)
        loss = terms["loss"]
        aux_loss, aux_iou = mask_head_loss(self.model._aux["dyn_logits"], self.model._aux["dyn_target"])
        if aux_loss is not None:
            loss = loss + self.loss_fn.dyn_mask_weight * aux_loss

        opt, sch = self.optimizers(), self.lr_schedulers()
        opt.zero_grad(set_to_none=True)
        nonfinite = (~torch.isfinite(loss)).long().to(loss.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(nonfinite, op=torch.distributed.ReduceOp.MAX)
        if nonfinite.item() > 0:
            self.print(f"[step {self.global_step}] non-finite loss on some rank, skipping")
            return None
        self.manual_backward(loss)
        params = [p for p in self.model.parameters() if p.requires_grad]
        for p in params:
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
        clip = float(self.cfg.trainer.grad_clip_norm)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=clip)
        opt.step()
        sch.step()

        log = {"train/loss": loss.detach(), "train/l2_loss": terms["l2"].detach(), "train/psnr": terms["psnr"],
               "train/perceptual_loss": terms["perceptual"].detach(), "train/dyn_perceptual_loss": terms["dyn_perceptual"].detach(),
               "train/dyn_grad_loss": terms["dyn_grad"].detach(), "train/lr": sch.get_last_lr()[0]}
        rp = self._region_psnrs(output[:, K:].detach(), images[:, K:], dyn_mask, K)
        for name in ("dyn", "static"):
            if math.isfinite(rp[name]):
                log[f"train/{name}_psnr"] = rp[name]
        if aux_loss is not None:
            log["train/dyn_mask_aux_loss"] = aux_loss.detach()
            log["train/dyn_mask_iou"] = aux_iou
        self.log_dict(log, prog_bar=True, on_step=True, on_epoch=False)
        if is_vis:
            self._log_grid(images, output, K, "train", tau=tau, image_ids=image_ids, dyn_mask=dyn_mask)

    def _pred_cond_mask(self, H, W):
        aux = self.model._aux
        if aux is None:
            return None
        pm = torch.sigmoid(aux["dyn_logits"][0].detach().float()).unsqueeze(1)
        return torch.nn.functional.interpolate(pm, size=(H, W), mode="nearest").cpu()

    def _save_vis(self, grid, name):
        vis_dir = os.path.join(self.cfg.experiment_path, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        Image.fromarray(grid).save(os.path.join(vis_dir, f"{name}_step{self.global_step:07d}.png"))

    def _log_grid(self, images, output, K, prefix, tau=None, image_ids=None, dyn_mask=None, grid_sizes=None):
        """Batch element 0 -> N x N grid template (grid_sizes given) or a row-based compare strip."""
        if self.global_rank != 0:
            return
        H, W = images.shape[-2:]
        mask0 = dyn_mask[0].detach().float().cpu() if (dyn_mask is not None and float(dyn_mask.abs().sum()) > 0) else None
        static = self.model._static_render
        static0 = static[0].detach().float().cpu() if static is not None else None
        if grid_sizes is not None:
            grid = make_grid_vis(images[0].detach().float().cpu(), output[0].detach().float().cpu(), K, *grid_sizes,
                                 static_pred=static0, gt_mask=mask0, pred_mask=self._pred_cond_mask(H, W),
                                 title=f"@step{self.global_step}")
        else:
            grid = make_compare_grid(images[0].detach().float().cpu(), output[0].detach().float().cpu(), K,
                                     tau=tau[0].detach().cpu() if tau is not None else None,
                                     image_ids=image_ids[0] if image_ids is not None else None,
                                     dyn_mask=mask0, pred_mask=self._pred_cond_mask(H, W), static_pred=static0)
        self._save_vis(grid, prefix)

    def _render_chunked(self, images, rays, cam_token, tau, dyn_mask, K, chunk=6, want_static=False):
        """Render many targets from the same conditioning in chunks; returns (full output, static-only output or None)."""
        n_tgt = images.shape[1] - K
        out, out_static = images.clone(), images.clone()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, n_tgt, chunk):
                idx = torch.arange(s, min(s + chunk, n_tgt), device=images.device)
                sel = torch.cat([torch.arange(K, device=images.device), K + idx])
                o = self.model(images[:, sel], rays[:, sel], cam_token[:, sel], num_cond_views=K, tau=tau[:, sel],
                               dyn_mask=dyn_mask[:, sel], return_static_render=want_static)
                out[:, K + idx] = o[:, K:].to(out.dtype)
                if want_static:
                    out_static[:, K + idx] = self.model._static_render[:, K:].to(out_static.dtype)
        return out, (out_static if want_static else None)

    @staticmethod
    def _region_psnrs(pred, target, dyn_mask, K):
        err = ((pred.float() - target.float()) ** 2).mean(dim=2)
        p = lambda x: float(-10.0 * torch.log10(x.clamp(min=1e-10)))
        out = {"full": p(err.mean()), "dyn": float("nan"), "static": float("nan")}
        m = dyn_mask[:, K:, 0].float() > 0.5
        if m.any():
            out["dyn"] = p(err[m].mean())
        if (~m).any():
            out["static"] = p(err[~m].mean())
        return out

    def _eval_step_nvidia(self, batch, prefix, batch_idx):
        import re
        images, rays, image_ids, cam_token, tau, dyn_mask, K = batch
        K = int(K)
        out, out_static = self._render_chunked(images, rays, cam_token, tau, dyn_mask, K, want_static=True)
        pred, target = out[:, K:], images[:, K:]
        B, Vt = pred.shape[:2]
        H, W = images.shape[-2:]
        ct = lambda s: tuple(int(x) for x in re.search(r"cam(\d+)_t(\d+)", str(s)).groups())
        per_cam, all_m = {}, {"psnr": [], "ssim": [], "lpips": []}
        for b in range(B):
            for v in range(Vt):
                cam, _ = ct(image_ids[b][K + v])
                mm = self._metrics.compute_all(pred[b, v], target[b, v])
                for k in all_m:
                    all_m[k].append(mm[k])
                    per_cam.setdefault(cam, {"psnr": [], "ssim": [], "lpips": []})[k].append(mm[k])
        payload = {f"{prefix}/{k}": sum(v) / max(1, len(v)) for k, v in all_m.items()}
        for cam, d in per_cam.items():
            payload.update({f"{prefix}/cam{cam:02d}_{k}": sum(v) / max(1, len(v)) for k, v in d.items()})
        self.log_dict(payload, on_epoch=True, sync_dist=True)
        if self.global_rank != 0:
            return
        cams = []
        for v in range(Vt):
            c = ct(image_ids[0][K + v])[0]
            if c not in cams:
                cams.append(c)
        C, T = len(cams), Vt // len(cams)
        rs = lambda x: x[0, K:].detach().float().cpu().reshape(C, T, 3, H, W)
        scene = re.match(r"nvidia_(.+)_cam\d+_t\d+", str(image_ids[0][0])).group(1)
        grid = make_nvidia_grid(images[0, :K].detach().float().cpu(), rs(images), rs(out), rs(out_static),
                                dyn_mask[0].detach().float().cpu(), self._pred_cond_mask(H, W),
                                [f"cam{c:02d}" for c in cams], list(range(T)), title=f"{scene}@step{self.global_step}")
        self._save_vis(grid, f"{prefix}_{scene}")

    def _eval_step(self, batch, prefix, batch_idx):
        images, rays, image_ids, cam_token, tau, dyn_mask, K = batch
        K = int(K)
        if self._metrics is None:
            self._metrics = MetricsComputer(self.device)
        if any(str(s).startswith("nvidia_") for s in image_ids[0]):
            return self._eval_step_nvidia(batch, prefix, batch_idx)
        n_views = images.shape[1]
        is_grid = n_views == K * K + 2 * K - 1
        want_vis = batch_idx == 0
        if is_grid:
            pred, static_full = self._render_chunked(images, rays, cam_token, tau, dyn_mask, K, want_static=want_vis)
            self.model._static_render = static_full
        else:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = self.model(images, rays, cam_token, num_cond_views=K, tau=tau, dyn_mask=dyn_mask, return_static_render=want_vis)
        target, pred_t = images[:, K:], pred[:, K:]
        Vt = pred_t.shape[1]
        m_all = {"psnr": [], "ssim": [], "lpips": []}
        psnr_v = [0.0] * Vt
        psnr_known, psnr_novel = [], []
        for b in range(pred_t.shape[0]):
            cond_times = set(tau[b, :K].round().tolist())
            for v in range(Vt):
                m = self._metrics.compute_all(pred_t[b, v], target[b, v])
                for k in m_all:
                    m_all[k].append(m[k])
                psnr_v[v] += m["psnr"] / pred_t.shape[0]
                if not is_grid:
                    (psnr_known if round(float(tau[b, K + v])) in cond_times else psnr_novel).append(m["psnr"])
        payload = {f"{prefix}/{k}": sum(v) / max(1, len(v)) for k, v in m_all.items()}
        if is_grid:
            payload[f"{prefix}/self_psnr"] = sum(psnr_v[:K]) / K
            payload[f"{prefix}/others_psnr"] = sum(psnr_v[K:K * K]) / (K * K - K)
            payload[f"{prefix}/novel_time_psnr"] = sum(psnr_v[K * K:]) / max(1, Vt - K * K)
        if psnr_known:
            payload[f"{prefix}/known_psnr"] = sum(psnr_known) / len(psnr_known)
        if psnr_novel:
            payload[f"{prefix}/novel_time_psnr"] = sum(psnr_novel) / len(psnr_novel)
        rp = self._region_psnrs(pred_t, target, dyn_mask, K)
        for name in ("dyn", "static"):
            if math.isfinite(rp[name]):
                payload[f"{prefix}/{name}_psnr"] = rp[name]
        _, iou = mask_head_loss(self.model._aux["dyn_logits"], self.model._aux["dyn_target"])
        if iou is not None:
            payload[f"{prefix}/dyn_mask_iou"] = float(iou)
        self.log_dict(payload, prog_bar=True, on_epoch=True, sync_dist=True)
        if want_vis:
            gsz = (K, K * (K - 1), n_views - K - K - K * (K - 1)) if is_grid else None
            self._log_grid(images, pred, K, prefix, tau=tau, image_ids=None if is_grid else image_ids,
                           dyn_mask=dyn_mask, grid_sizes=gsz)

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        names = self.trainer.datamodule.val_names
        self._eval_step(batch, prefix=names[dataloader_idx], batch_idx=batch_idx)

    def configure_optimizers(self):
        decay, nodecay = split_decay_params(self.model)
        opt_cfg = self.cfg.trainer.optimizer
        optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": opt_cfg.weight_decay},
                                       {"params": nodecay, "weight_decay": 0.0}],
                                      lr=opt_cfg.lr, betas=tuple(opt_cfg.betas), fused=True)
        lr_lambda = get_lr_lambda(self.cfg.trainer.scheduler.name, self.cfg.trainer.max_steps, self.cfg.trainer.scheduler)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    if not is_rank_zero:
        sys.stdout = open(os.devnull, "w")
    OmegaConf.set_struct(cfg, False)
    cfg.experiment_path = os.environ.setdefault("FL4N_EXP_PATH", str(cfg.experiment_path))
    Path(cfg.experiment_path).mkdir(exist_ok=True, parents=True)
    if is_rank_zero:
        with open(os.path.join(cfg.experiment_path, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
    L.seed_everything(int(cfg.get("seed", 0)), workers=True)

    model = FL4NModule(cfg)
    datamodule = FL4NDataModule(data_cfg=cfg.data)
    ckpt_path = cfg.get("ckpt_path") or None
    save_path = os.path.join(cfg.experiment_path, "checkpoints")

    if cfg.mode == "train":
        logger = CSVLogger(save_dir=cfg.experiment_path, name="", version="")
        checkpoint_callback = ModelCheckpoint(dirpath=save_path, filename="step={step}", auto_insert_metric_name=False,
                                              save_last=True, save_top_k=-1, every_n_train_steps=int(cfg.ckpt_every_n_steps))
        trainer = L.Trainer(
            strategy=DDPStrategy(find_unused_parameters=bool(cfg.find_unused_parameters), gradient_as_bucket_view=True,
                                 timeout=datetime.timedelta(minutes=int(cfg.get("ddp_timeout_min", 60)))),
            devices=cfg.devices, num_nodes=cfg.num_nodes, logger=logger, num_sanity_val_steps=0, precision=cfg.precision,
            val_check_interval=cfg.val_check_interval, check_val_every_n_epoch=None, limit_val_batches=cfg.limit_val_batches,
            log_every_n_steps=cfg.log_every_n_steps, max_steps=cfg.trainer.max_steps, sync_batchnorm=True,
            callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        )
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif cfg.mode == "eval":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        trainer = L.Trainer(strategy="ddp", devices=cfg.devices, num_nodes=cfg.num_nodes, logger=False,
                            precision=cfg.precision, limit_val_batches=cfg.limit_val_batches)
        results = trainer.validate(model, datamodule=datamodule, ckpt_path=ckpt_path)
        if is_rank_zero and cfg.get("eval_out"):
            with open(cfg.eval_out, "w") as f:
                json.dump({"ckpt_path": ckpt_path, "results": results}, f, indent=2)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")


if __name__ == "__main__":
    main()
