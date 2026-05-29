"""
Pre-training trainer for Multi-resolution HuBERT.

Implements the self-supervised masked-unit-prediction objective described in
the diagram.  Key features:

  • Mixed precision via ``torch.cuda.amp`` (GradScaler + autocast)
  • Step-based linear-warmup → cosine-decay schedule
  • Per-step logging (loss, per-head accuracy, gradient norm, LR) to
    wandb / TensorBoard via ``TrainLogger``
  • Checkpoint save every N steps + at epoch end; resume supported
  • Separate tracking of high-res (H₃) and low-res (H₂) prediction stats
"""

from __future__ import annotations

import logging
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .losses import MultiResPretrainingLoss
from .scheduler import get_warmup_cosine_schedule
from .logger import TrainLogger
from .local_logger import LocalRunLogger
from .checkpoint_syncer import CheckpointSyncer

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PretrainConfig:
    output_dir:   str   = "checkpoints/pretrain"
    epochs:       int   = 100
    learning_rate: float = 1e-4
    weight_decay:  float = 1e-2
    warmup_steps:  int   = 10_000
    grad_clip:     float = 1.0
    use_amp:       bool  = True
    log_every:              int   = 100
    save_every:             int   = 5_000  # step checkpoints; ignored when save_best_only=True
    keep_last_n_checkpoints: int  = 3
    save_best_only:         bool  = True   # only write a checkpoint when avg epoch loss improves
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss weights for g^q_R1 (hi) and g^q_R2 (lo)
    hi_loss_weight: float = 1.0
    lo_loss_weight: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _masked_accuracy(
    logits:  Tensor,   # (B, T, V)
    targets: Tensor,   # (B, T)  int64, -1 = pad / ignored
    mask:    Tensor,   # (B, T)  bool,  True = masked position
) -> float:
    """Fraction of correctly predicted cluster IDs at masked, non-padding positions."""
    T = min(logits.shape[1], targets.shape[1], mask.shape[1])
    logits  = logits[:, :T]
    targets = targets[:, :T]
    mask    = mask[:, :T]
    valid = mask & (targets >= 0)
    if not valid.any():
        return float("nan")
    pred = logits[valid].argmax(dim=-1)
    return (pred == targets[valid]).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Pre-training trainer
# ─────────────────────────────────────────────────────────────────────────────

class PretrainTrainer:
    """
    Orchestrates HuBERT-style masked unit-prediction pre-training.

    Args:
        model:        ``MultiResHuBERT`` initialised with ``pretrain=True``.
        train_loader: Produced by ``PretrainDataset`` + ``pretrain_collate_fn``.
        config:       ``PretrainConfig`` instance.
        logger:       ``TrainLogger`` (wandb / TensorBoard).  Pass ``None``
                      to log only to stdout.
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        config:       PretrainConfig,
        logger:       Optional[TrainLogger]    = None,
        local_logger: Optional[LocalRunLogger] = None,
        syncer:       Optional[CheckpointSyncer] = None,
    ) -> None:
        self.cfg          = config
        self.device       = torch.device(config.device)
        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.logger       = logger or TrainLogger()
        self.local_logger = local_logger
        self.syncer       = syncer

        self.criterion = MultiResPretrainingLoss(
            hi_weight         = config.hi_loss_weight,
            lo_weight         = config.lo_loss_weight,
            downsample_stride = getattr(model, "down", None)
                                and model.down.stride or 2,
        )

        self.optimizer = AdamW(
            model.trainable_parameters()
            if hasattr(model, "trainable_parameters")
            else model.parameters(),
            lr           = config.learning_rate,
            weight_decay = config.weight_decay,
            betas        = (0.9, 0.98),
            eps          = 1e-6,
        )

        # Total steps = epochs × steps_per_epoch
        steps_per_epoch   = len(train_loader)
        self.total_steps  = config.epochs * steps_per_epoch
        self.scheduler    = get_warmup_cosine_schedule(
            self.optimizer,
            warmup_steps = config.warmup_steps,
            total_steps  = self.total_steps,
            min_lr_ratio = 0.0,
        )

        # Wrap with DataParallel when multiple CUDA GPUs are available.
        # Optimizer and criterion are built first so they reference the raw
        # model's parameters before wrapping.
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            log.info(
                "DataParallel: using %d GPUs.",
                torch.cuda.device_count(),
            )

        # AMP
        self.scaler = GradScaler(enabled=config.use_amp and torch.cuda.is_available())

        # State
        self.global_step  = 0
        self.start_epoch  = 0
        self.output_dir   = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._epoch_ckpts: list = []   # rolling list of saved epoch checkpoint paths
        self._step_ckpts:  list = []   # rolling list of saved step checkpoint paths
        self.best_loss:    float = float("inf")

    # ──────────────────────────────────────────────────────────────────────
    # Single optimiser step
    # ──────────────────────────────────────────────────────────────────────

    def _step(
        self,
        batch: Dict[str, Tensor | List[str]],
    ) -> Dict[str, float]:
        """
        Executes one forward–backward–optimiser step.

        Returns a metrics dict with keys:
          loss, loss_hi, loss_lo, acc_hi, acc_lo, lr, grad_norm
        """
        waveforms   = batch["waveforms"].to(self.device)        # (B, T)
        attn_mask   = batch["attention_mask"].to(self.device)   # (B, T)
        hi_targets  = batch["hi_targets"].to(self.device)       # (B, F)
        lo_targets  = batch["lo_targets"].to(self.device)       # (B, F')

        n_feat_frames = waveforms.numel() // 320   # approx feature frames in batch
        t0 = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)

        # ── forward + loss (inside autocast) ─────────────────────────────
        use_autocast = self.cfg.use_amp and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=use_autocast):
            out = self.model(waveforms, attn_mask, apply_mask=True)

            # Align target sequence lengths with model output lengths
            T_hi = out.logits_hi.shape[1]
            T_lo = out.logits_lo.shape[1]

            hi_t  = hi_targets[:, :T_hi]
            lo_t  = lo_targets[:, :T_lo]
            mask  = out.mask_ids[:, :T_hi]   # (B, T_hi) bool

            # Per-head losses (for individual logging)
            loss_hi = self.criterion._masked_ce(out.logits_hi, hi_t, mask)
            # Downsample mask for low-res head
            B = mask.shape[0]
            m      = self._unwrap()
            stride = getattr(m, "down", None) and m.down.stride or 2
            mask_lo = (
                F.max_pool1d(
                    mask.float().unsqueeze(1),
                    kernel_size=stride, stride=stride,
                    padding=0, ceil_mode=True,
                ).squeeze(1)[:, :T_lo] > 0.5
            )
            loss_lo = self.criterion._masked_ce(out.logits_lo, lo_t, mask_lo)

            loss = (
                self.cfg.hi_loss_weight * loss_hi
                + self.cfg.lo_loss_weight * loss_lo
            )

        # ── backward ─────────────────────────────────────────────────────
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip
        ).item()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1

        # ── accuracy on masked positions ──────────────────────────────────
        acc_hi = _masked_accuracy(out.logits_hi.detach(), hi_t, mask)
        acc_lo = _masked_accuracy(out.logits_lo.detach(), lo_t, mask_lo)

        current_lr      = self.scheduler.get_last_lr()[0]
        step_secs       = time.perf_counter() - t0
        tokens_per_sec  = n_feat_frames / max(step_secs, 1e-6)

        return {
            "train/loss":        loss.item(),
            "train/loss_hi":     loss_hi.item(),
            "train/loss_lo":     loss_lo.item(),
            "train/acc_hi":      acc_hi,
            "train/acc_lo":      acc_lo,
            "train/lr":          current_lr,
            "train/grad_norm":   grad_norm,
            "train/tok_per_sec": tokens_per_sec,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Epoch loop
    # ──────────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        running: Dict[str, float] = {}
        n_steps = 0

        for batch in self.train_loader:
            metrics = self._step(batch)

            # Accumulate running averages
            for k, v in metrics.items():
                if math.isfinite(v):
                    running[k] = running.get(k, 0.0) + v
            n_steps += 1

            # ── per-step logging ─────────────────────────────────────────
            if self.global_step % self.cfg.log_every == 0:
                self.logger.log(metrics, step=self.global_step)
                if self.local_logger is not None:
                    self.local_logger.log_step(
                        step          = self.global_step,
                        epoch         = epoch + 1,
                        total_epochs  = self.cfg.epochs,
                        metrics       = metrics,
                        tokens_per_sec= metrics.get("train/tok_per_sec"),
                    )

            # ── periodic step checkpoint (skipped in save_best_only mode) ────
            if (not self.cfg.save_best_only
                    and self.global_step % self.cfg.save_every == 0):
                step_name = f"step_{self.global_step:07d}.pt"
                self._save(step_name, epoch=epoch)
                self._step_ckpts.append(step_name)
                keep = self.cfg.keep_last_n_checkpoints
                while len(self._step_ckpts) > keep:
                    self._prune_ckpt(self._step_ckpts.pop(0))

        # Return epoch-averaged metrics
        return {k: v / max(1, n_steps) for k, v in running.items()}

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        log.info(
            "Pre-training on %s | epochs=%d | warmup=%d steps | total=%d steps",
            self.device, self.cfg.epochs, self.cfg.warmup_steps, self.total_steps,
        )

        for epoch in range(self.start_epoch, self.cfg.epochs):
            log.info("── Epoch %d/%d ──", epoch + 1, self.cfg.epochs)
            epoch_metrics = self._train_epoch(epoch)

            # Log epoch-level averages
            epoch_summary = {f"epoch/{k.split('/')[1]}": v
                             for k, v in epoch_metrics.items()}
            self.logger.log(epoch_summary, step=self.global_step)

            log.info(
                "Epoch %d done | avg loss %.4f | acc_hi %.3f | acc_lo %.3f",
                epoch + 1,
                epoch_metrics.get("train/loss", float("nan")),
                epoch_metrics.get("train/acc_hi", float("nan")),
                epoch_metrics.get("train/acc_lo", float("nan")),
            )
            if self.local_logger is not None:
                self.local_logger.log_epoch(
                    epoch        = epoch + 1,
                    total_epochs = self.cfg.epochs,
                    step         = self.global_step,
                    metrics      = epoch_metrics,
                )
                # Push updated logs (metrics.csv, train.log) to Google Drive
                if self.syncer:
                    log_remote = self.syncer._cfg.get("log_remote", "")
                    if log_remote:
                        self.syncer.sync_dir_async(self.local_logger.run_dir, log_remote)

            # ── Checkpoint save ───────────────────────────────────────────────
            current_loss = epoch_metrics.get("train/loss", float("inf"))
            if self.cfg.save_best_only:
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self._save("best_model.pt", epoch=epoch + 1)
                    log.info("New best loss %.4f → saved best_model.pt", current_loss)
                else:
                    log.info(
                        "Loss %.4f ≥ best %.4f — checkpoint skipped.",
                        current_loss, self.best_loss,
                    )
            else:
                ckpt_name = f"epoch_{epoch + 1:03d}.pt"
                self._save(ckpt_name, epoch=epoch + 1)
                self._epoch_ckpts.append(ckpt_name)
                keep = self.cfg.keep_last_n_checkpoints
                while len(self._epoch_ckpts) > keep:
                    self._prune_ckpt(self._epoch_ckpts.pop(0))

        # Final checkpoint
        self._save("final_pretrain.pt", epoch=self.cfg.epochs)
        self.logger.finish()
        if self.local_logger is not None:
            self.local_logger.close()
        if self.syncer:
            self.syncer.wait_all()
        log.info("Pre-training complete. Checkpoint → %s", self.output_dir)

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ──────────────────────────────────────────────────────────────────────

    def _unwrap(self) -> nn.Module:
        """Return the raw model, unwrapping DataParallel if present."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    @property
    def _remote_mode(self) -> bool:
        """True when checkpoints go directly to Google Drive (no local write)."""
        return bool(
            self.syncer
            and self.syncer.enabled
            and self.syncer._cfg.get("method") == "rclone"
        )

    def _save(self, name: str, epoch: Optional[int] = None) -> None:
        state = {
            "model_state":     self._unwrap().state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "global_step":     self.global_step,
            "epoch":           epoch if epoch is not None else self.start_epoch,
            "best_loss":       self.best_loss,
            "config":          self.cfg.__dict__,
            "run_dir": str(self.local_logger.run_dir)
                       if self.local_logger else None,
        }

        if self._remote_mode:
            # Write to /tmp (separate from the full /workspace disk),
            # upload to Google Drive synchronously, then delete the temp file.
            tmp = Path(tempfile.gettempdir()) / name
            torch.save(state, tmp)
            log.debug("Serialised to %s, uploading …", tmp)
            self.syncer.upload_and_delete(tmp, name)
        else:
            path = self.output_dir / name
            torch.save(state, path)
            log.debug("Checkpoint saved → %s", path)
            if self.syncer:
                self.syncer.sync_async(path)

    def _prune_ckpt(self, name: str) -> None:
        """Delete an old checkpoint — from Drive in remote mode, from disk otherwise."""
        if self._remote_mode:
            self.syncer.delete_remote_checkpoint(name)
        else:
            path = self.output_dir / name
            if path.exists():
                path.unlink()
                log.debug("Removed old checkpoint: %s", name)

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint (local file or Google Drive name)."""
        p = Path(path)
        tmp_downloaded: Optional[Path] = None

        if not p.exists() and self._remote_mode:
            # Checkpoint is on Drive — download to /tmp first
            log.info("Local file not found; downloading %s from Drive …", p.name)
            tmp_dir = Path(tempfile.gettempdir())
            tmp_downloaded = self.syncer.download_checkpoint(p.name, tmp_dir)
            if tmp_downloaded is None:
                raise FileNotFoundError(
                    f"Checkpoint '{p.name}' not found locally or on Google Drive."
                )
            p = tmp_downloaded

        ckpt = torch.load(p, map_location=self.device)
        self._unwrap().load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.global_step = ckpt["global_step"]
        self.start_epoch = ckpt.get("epoch", 0)
        self.best_loss   = ckpt.get("best_loss", float("inf"))

        if tmp_downloaded is not None:
            tmp_downloaded.unlink(missing_ok=True)

        log.info(
            "Resumed from %s  (step=%d, epoch=%d)",
            path, self.global_step, self.start_epoch,
        )
