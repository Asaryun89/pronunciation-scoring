"""
Fine-tuning trainer for MultiResHuBERT (Speechocean762 pronunciation assessment).

Features added over the naive baseline:
  • Mixed precision via ``torch.cuda.amp`` (GradScaler + autocast)
  • Step-based linear-warmup → cosine-decay LR schedule
  • Per-step and per-epoch logging via ``TrainLogger`` (wandb / TensorBoard)
  • Checkpoint save on best val loss + at end of training; full resume support
  • Per-dimension PCC and MSE reported on the validation set after every epoch

Usage
─────
    from training.trainer import Trainer, TrainerConfig
    cfg     = TrainerConfig(...)
    trainer = Trainer(model, train_loader, val_loader, cfg, logger)
    trainer.train()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .losses import PronunciationLoss
from .scheduler import get_warmup_cosine_schedule
from .logger import TrainLogger
from .local_logger import LocalRunLogger
from .checkpoint_syncer import CheckpointSyncer

log = logging.getLogger(__name__)

SCORE_KEYS = ["accuracy", "fluency", "completeness", "prosodic", "total"]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    output_dir:    str   = "checkpoints"
    epochs:        int   = 30
    learning_rate: float = 1e-4
    weight_decay:  float = 1e-2
    warmup_steps:  int   = 500
    grad_clip:     float = 1.0
    use_amp:       bool  = True
    log_every:     int   = 50
    eval_every:    int   = 1
    save_best:     bool  = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    score_weights: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 2.0]
    )
    mse_weight:  float = 1.0
    corr_weight: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        config:       TrainerConfig,
        logger:       Optional[TrainLogger]      = None,
        local_logger: Optional[LocalRunLogger]   = None,
        syncer:       Optional[CheckpointSyncer] = None,
    ) -> None:
        self.cfg          = config
        self.device       = torch.device(config.device)
        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.logger       = logger or TrainLogger()
        self.local_logger = local_logger
        self.syncer       = syncer

        self.criterion = PronunciationLoss(
            score_weights = config.score_weights,
            mse_weight    = config.mse_weight,
            corr_weight   = config.corr_weight,
        ).to(self.device)

        self.optimizer = AdamW(
            model.trainable_parameters()
            if hasattr(model, "trainable_parameters")
            else model.parameters(),
            lr           = config.learning_rate,
            weight_decay = config.weight_decay,
        )

        total_steps = config.epochs * len(train_loader)
        self.scheduler = get_warmup_cosine_schedule(
            self.optimizer,
            warmup_steps = config.warmup_steps,
            total_steps  = total_steps,
            min_lr_ratio = 0.0,
        )

        self.scaler = GradScaler(
            enabled = config.use_amp and torch.cuda.is_available()
        )

        self.output_dir    = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.global_step   = 0
        self.start_epoch   = 0

    # ──────────────────────────────────────────────────────────────────────
    # Single step
    # ──────────────────────────────────────────────────────────────────────

    def _step(
        self, batch: dict
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, float]]:
        """Forward + backward.  Returns (loss, preds, labels, step_metrics)."""
        waveforms = batch["waveforms"].to(self.device)
        masks     = batch["attention_mask"].to(self.device)
        labels    = batch["labels"].to(self.device)

        t0 = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)

        use_ac = self.cfg.use_amp and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=use_ac):
            out  = self.model(waveforms, masks, apply_mask=False)
            loss = self.criterion(out.scores, labels)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip
        ).item()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1

        step_secs = time.perf_counter() - t0
        step_metrics = {
            "train/loss":      loss.item(),
            "train/lr":        self.scheduler.get_last_lr()[0],
            "train/grad_norm": grad_norm,
            "train/tok_per_sec": waveforms.shape[0] / max(step_secs, 1e-6),
        }
        return loss.detach(), out.scores.detach(), labels, step_metrics

    # ──────────────────────────────────────────────────────────────────────
    # Train / evaluate
    # ──────────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            loss, preds, labels, step_metrics = self._step(batch)
            total_loss += loss.item()

            if self.global_step % self.cfg.log_every == 0:
                self.logger.log(step_metrics, step=self.global_step)
                if self.local_logger is not None:
                    self.local_logger.log_step(
                        step          = self.global_step,
                        epoch         = epoch + 1,
                        total_epochs  = self.cfg.epochs,
                        metrics       = step_metrics,
                        tokens_per_sec= step_metrics.get("train/tok_per_sec"),
                    )

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        """
        Returns:
            val_loss:  combined PronunciationLoss scalar
            metrics:   ``{pcc_<dim>: float, mse_<dim>: float}`` for each of
                       the 5 MOS dimensions
        """
        self.model.eval()
        total_loss = 0.0
        all_preds:  List[Tensor] = []
        all_labels: List[Tensor] = []

        use_ac = self.cfg.use_amp and self.device.type == "cuda"
        for batch in self.val_loader:
            waveforms = batch["waveforms"].to(self.device)
            masks     = batch["attention_mask"].to(self.device)
            labels    = batch["labels"].to(self.device)

            with torch.autocast(device_type=self.device.type, enabled=use_ac):
                out  = self.model(waveforms, masks, apply_mask=False)
                loss = self.criterion(out.scores, labels)

            total_loss += loss.item()
            all_preds.append(out.scores.cpu().float())
            all_labels.append(labels.cpu().float())

        p = torch.cat(all_preds,  dim=0).numpy()   # (N, 5)
        t = torch.cat(all_labels, dim=0).numpy()   # (N, 5)

        metrics: Dict[str, float] = {}
        for i, key in enumerate(SCORE_KEYS):
            r, _  = pearsonr(p[:, i], t[:, i])
            mse   = float(np.mean((p[:, i] - t[:, i]) ** 2))
            metrics[f"val/pcc_{key}"] = float(r)
            metrics[f"val/mse_{key}"] = mse

        return total_loss / len(self.val_loader), metrics

    # ──────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        log.info("Fine-tuning on %s  (epochs=%d)", self.device, self.cfg.epochs)

        for epoch in range(self.start_epoch, self.cfg.epochs):
            train_loss = self._train_epoch(epoch)

            if (epoch + 1) % self.cfg.eval_every == 0:
                val_loss, metrics = self.evaluate()

                pcc_total = metrics.get("val/pcc_total", float("nan"))
                mse_total = metrics.get("val/mse_total", float("nan"))

                log.info(
                    "Epoch %d/%d — train %.4f  val %.4f  "
                    "PCC(total)=%.4f  MSE(total)=%.4f",
                    epoch + 1, self.cfg.epochs,
                    train_loss, val_loss, pcc_total, mse_total,
                )

                epoch_metrics = {
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss":   val_loss,
                    **metrics,
                }
                self.logger.log(epoch_metrics, step=self.global_step)

                if self.local_logger is not None:
                    self.local_logger.log_epoch(
                        epoch        = epoch + 1,
                        total_epochs = self.cfg.epochs,
                        step         = self.global_step,
                        metrics      = epoch_metrics,
                    )

                if self.cfg.save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save("best_model.pt", val_loss, metrics)
                    log.info("  → new best saved (val %.4f)", val_loss)

        self._save("final_model.pt", self.best_val_loss, {})
        self.logger.finish()
        if self.local_logger is not None:
            self.local_logger.close()
        if self.syncer:
            self.syncer.wait_all()
        log.info("Training complete. Best val loss: %.4f", self.best_val_loss)

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ──────────────────────────────────────────────────────────────────────

    def _save(self, name: str, val_loss: float, metrics: Dict[str, float]) -> None:
        path = self.output_dir / name
        torch.save(
            {
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state":    self.scaler.state_dict(),
                "best_val_loss":   self.best_val_loss,
                "global_step":     self.global_step,
                "epoch":           self.start_epoch,
                "metrics":         metrics,
                "run_dir": str(self.local_logger.run_dir)
                           if self.local_logger else None,
            },
            path,
        )
        if self.syncer:
            self.syncer.sync_async(path)

    def load_checkpoint(self, path: str) -> None:
        """Resume fine-tuning from a saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.global_step   = ckpt.get("global_step", 0)
        self.start_epoch   = ckpt.get("epoch", 0)
        log.info(
            "Resumed from %s  (step=%d, best_val=%.4f)",
            path, self.global_step, self.best_val_loss,
        )
