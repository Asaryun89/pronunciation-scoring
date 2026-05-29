"""
Fusion-C fine-tuning trainer for Multi-resolution HuBERT on Speechocean762.

Entry point:
    cd multi_res_hubert
    python training/finetune_fusionC.py --config configs/finetune_config.yaml
    python training/finetune_fusionC.py --config configs/finetune_config.yaml --resume checkpoints/finetune/last.pt

Outputs per run:
    logs/finetune/run_YYYYMMDD_HHMMSS/
    ├── train_steps.csv      step-level: step, epoch, loss, lr, grad_norm, elapsed_sec
    ├── val_epochs.csv       epoch-level: all PCC + MSE dims
    └── config_snapshot.yaml copy of the config used

Checkpoints:
    checkpoints/finetune/
    ├── best_model.pt         best val_pcc_total ever seen
    └── last_epoch_NNN.pt     rolling last-N checkpoint
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

# Allow `python training/finetune_fusionC.py` from multi_res_hubert/ root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.speechocean_dataset import (
    FUSION_SCORE_KEYS,
    SpeechoceanFusionDataset,
    fusion_collate_fn,
)
from model.multireshubert_finetune import MultiResHuBERTFinetune
from training.scheduler import get_warmup_cosine_schedule
from training.checkpoint_syncer import CheckpointSyncer, build_syncer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def _pearson_r(pred: Tensor, target: Tensor) -> Tensor:
    """
    Per-dimension Pearson correlation coefficient.

    Args:
        pred, target: [B, D]

    Returns:
        r: [D]  — correlation per score dimension
    """
    vp  = pred   - pred.mean(0, keepdim=True)
    vt  = target - target.mean(0, keepdim=True)
    num = (vp * vt).sum(0)
    den = vp.norm(dim=0) * vt.norm(dim=0) + 1e-8
    return num / den


def pcc_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Mean over score dimensions of  (1 − Pearson r).

    Args:
        pred, target: [B, n_scores]

    Returns:
        Scalar loss.
    """
    return (1.0 - _pearson_r(pred, target)).mean()


def combined_loss(
    pred:       Tensor,
    target:     Tensor,
    mse_weight: float,
    pcc_weight: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Returns (total_loss, mse_term, pcc_term)."""
    mse  = F.mse_loss(pred, target)
    pcc  = pcc_loss(pred, target)
    return mse_weight * mse + pcc_weight * pcc, mse, pcc


# ─────────────────────────────────────────────────────────────────────────────
# CSV / logging helpers
# ─────────────────────────────────────────────────────────────────────────────

_STEP_COLS  = ["step", "epoch", "loss", "mse", "pcc", "lr", "grad_norm", "elapsed_sec"]
_EPOCH_COLS = [
    "epoch", "step", "train_loss",
    "val_pcc_total", "val_pcc_accuracy", "val_pcc_fluency",
    "val_pcc_prosodic", "val_pcc_completeness",
    "val_mse_total", "lr", "elapsed_sec",
]


def _fmt(v: Optional[float], prec: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.{prec}f}"


class _CSVWriter:
    def __init__(self, path: Path, cols: List[str], append: bool = False) -> None:
        mode = "a" if append else "w"
        self._f   = open(path, mode, newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._f, fieldnames=cols, extrasaction="ignore")
        if not append:
            self._csv.writeheader()
            self._f.flush()

    def write(self, row: Dict) -> None:
        self._csv.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    """Load YAML and resolve relative paths anchored to the config file directory."""
    cfg_file = Path(path).resolve()
    cfg_dir  = cfg_file.parent

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _PATH_KEYS = {
        ("model",    "pretrain_checkpoint"),
        ("training", "output_dir"),
        ("logging",  "tb_log_dir"),
    }
    for section, key in _PATH_KEYS:
        if section not in cfg:
            continue
        raw = cfg[section].get(key)
        if raw and raw != "null" and not Path(raw).is_absolute():
            cfg[section][key] = str(cfg_dir / raw)

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class FusionCTrainer:
    """
    Training loop for Fusion-C pronunciation scoring.

    Args:
        model:        MultiResHuBERTFinetune (on CPU; moved to device internally).
        train_loader: DataLoader produced by SpeechoceanFusionDataset + fusion_collate_fn.
        val_loader:   Validation DataLoader.
        cfg:          Full parsed config dict.
        run_dir:      Timestamped run directory for logs.
    """

    def __init__(
        self,
        model:        MultiResHuBERTFinetune,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          dict,
        run_dir:      Path,
        syncer:       Optional[CheckpointSyncer] = None,
    ) -> None:
        tcfg = cfg["training"]
        lcfg = cfg.get("logging", {})

        self.cfg          = cfg
        self.tcfg         = tcfg
        self.device       = torch.device(tcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.run_dir      = run_dir
        self.output_dir   = Path(tcfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer + scheduler
        self.optimizer = model.get_optimizer()
        steps_per_epoch = len(train_loader)
        total_steps     = tcfg["epochs"] * steps_per_epoch
        self.scheduler  = get_warmup_cosine_schedule(
            self.optimizer,
            warmup_steps = tcfg.get("warmup_steps", 500),
            total_steps  = total_steps,
            min_lr_ratio = tcfg.get("min_lr", 1e-6) / max(
                tcfg["lr_speech_encoder"], tcfg["lr_fusion_head"]
            ),
        )

        self.scaler = GradScaler(
            enabled=tcfg.get("use_amp", True) and self.device.type == "cuda"
        )

        # Loss weights
        self.mse_w = tcfg.get("mse_weight", 0.5)
        self.pcc_w = tcfg.get("pcc_weight", 0.5)

        # Early stopping state
        self.syncer         = syncer
        self.best_metric    = float("-inf")
        self.patience_count = 0
        self.best_ckpt_path: Optional[Path] = None
        self._last_ckpts:    List[Path]      = []

        # State for resume
        self.global_step  = 0
        self.start_epoch  = 0

        # CSV loggers
        self._step_csv = _CSVWriter(run_dir / "train_steps.csv", _STEP_COLS)
        self._val_csv  = _CSVWriter(run_dir / "val_epochs.csv",  _EPOCH_COLS)
        self._t0       = time.monotonic()

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._t0, 1)

    def _elapsed_str(self) -> str:
        return str(timedelta(seconds=int(time.monotonic() - self._t0)))

    def _current_lr(self) -> float:
        # Report the speech-encoder group lr (group 0).
        return self.optimizer.param_groups[0]["lr"]

    # ──────────────────────────────────────────────────────────────────────
    # Single training step
    # ──────────────────────────────────────────────────────────────────────

    def _step(self, batch: Dict) -> Dict[str, float]:
        waveforms  = batch["waveforms"].to(self.device)
        attn_mask  = batch["attention_mask"].to(self.device)
        targets    = batch["labels"].to(self.device)
        transcripts = batch["transcripts"]

        self.optimizer.zero_grad(set_to_none=True)

        use_amp = self.tcfg.get("use_amp", True) and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            pred = self.model(waveforms, attn_mask, transcripts)
            loss, mse_t, pcc_t = combined_loss(
                pred, targets, self.mse_w, self.pcc_w
            )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.tcfg.get("grad_clip", 1.0)
        ).item()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1

        return {
            "loss":      loss.item(),
            "mse":       mse_t.item(),
            "pcc":       pcc_t.item(),
            "grad_norm": grad_norm,
            "lr":        self._current_lr(),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Epoch loops
    # ──────────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0

        for batch in self.train_loader:
            metrics = self._step(batch)
            total_loss += metrics["loss"]
            n += 1

            if self.global_step % self.tcfg.get("log_every", 20) == 0:
                log.info(
                    "[Ep %d | Step %d] loss=%.4f mse=%.4f pcc=%.4f "
                    "lr=%.2e gnorm=%.2f | %s",
                    epoch, self.global_step,
                    metrics["loss"], metrics["mse"], metrics["pcc"],
                    metrics["lr"], metrics["grad_norm"], self._elapsed_str(),
                )
                self._step_csv.write({
                    "step":        self.global_step,
                    "epoch":       epoch,
                    "loss":        _fmt(metrics["loss"]),
                    "mse":         _fmt(metrics["mse"]),
                    "pcc":         _fmt(metrics["pcc"]),
                    "lr":          _fmt(metrics["lr"], prec=8),
                    "grad_norm":   _fmt(metrics["grad_norm"]),
                    "elapsed_sec": self._elapsed(),
                })

        return total_loss / max(1, n)

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        all_pred:   List[np.ndarray] = []
        all_target: List[np.ndarray] = []

        for batch in self.val_loader:
            waveforms   = batch["waveforms"].to(self.device)
            attn_mask   = batch["attention_mask"].to(self.device)
            targets     = batch["labels"].to(self.device)
            transcripts = batch["transcripts"]

            pred = self.model(waveforms, attn_mask, transcripts)
            all_pred.append(pred.cpu().float().numpy())
            all_target.append(targets.cpu().float().numpy())

        preds   = np.concatenate(all_pred,   axis=0)   # [N, 5]
        targets = np.concatenate(all_target, axis=0)   # [N, 5]

        metrics: Dict[str, float] = {}
        for i, key in enumerate(FUSION_SCORE_KEYS):
            p, t   = preds[:, i], targets[:, i]
            corr   = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-6 else float("nan")
            metrics[f"val_pcc_{key}"] = corr
        metrics["val_mse_total"] = float(np.mean((preds - targets) ** 2))
        return metrics

    # ──────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────────

    def _save(self, name: str, epoch: int, val_metrics: Dict[str, float]) -> Path:
        path = self.output_dir / name
        torch.save(
            {
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state":    self.scaler.state_dict(),
                "global_step":     self.global_step,
                "epoch":           epoch,
                "best_metric":     self.best_metric,
                "val_metrics":     val_metrics,
                "config":          self.cfg,
                "run_dir":         str(self.run_dir),
            },
            path,
        )
        log.debug("Checkpoint → %s", path)
        if self.syncer:
            self.syncer.sync_async(path)
        return path

    def _manage_checkpoints(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        tcfg         = self.tcfg
        save_every   = tcfg.get("save_every_n_epochs", 5)
        keep_n       = tcfg.get("keep_last_n_checkpoints", 3)
        metric_key   = tcfg.get("best_metric", "val_pcc_total")
        min_delta    = tcfg.get("min_delta", 0.001)

        current = val_metrics.get(metric_key, float("-inf"))

        # Save best checkpoint when val_pcc_total improves.
        if current - self.best_metric > min_delta:
            self.best_metric = current
            self.patience_count = 0
            self.best_ckpt_path = self._save("best_model.pt", epoch, val_metrics)
            log.info("  ★ New best %s=%.4f → saved best_model.pt", metric_key, current)
        else:
            self.patience_count += 1

        # Rolling last-N checkpoint every save_every_n_epochs.
        if epoch % save_every == 0:
            name = f"last_epoch_{epoch:03d}.pt"
            p    = self._save(name, epoch, val_metrics)
            self._last_ckpts.append(p)
            # Prune oldest checkpoints beyond keep_n.
            while len(self._last_ckpts) > keep_n:
                old = self._last_ckpts.pop(0)
                if old.exists() and old != self.best_ckpt_path:
                    old.unlink()
                    log.debug("Removed old checkpoint: %s", old.name)

    # ──────────────────────────────────────────────────────────────────────
    # Main train loop
    # ──────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        tcfg         = self.tcfg
        epochs       = tcfg["epochs"]
        patience     = tcfg.get("patience", 10)
        early_stop   = tcfg.get("early_stopping", True)
        metric_key   = tcfg.get("best_metric", "val_pcc_total")

        log.info(
            "Fusion-C fine-tuning | device=%s | epochs=%d | "
            "lr_speech=%.1e | lr_fusion=%.1e",
            self.device, epochs,
            tcfg["lr_speech_encoder"], tcfg["lr_fusion_head"],
        )

        for epoch in range(self.start_epoch + 1, epochs + 1):
            log.info("── Epoch %d / %d ──", epoch, epochs)

            train_loss = self._train_epoch(epoch)
            val_metrics = self._validate()

            pcc_total = val_metrics.get("val_pcc_total", float("nan"))
            mse_total = val_metrics.get("val_mse_total", float("nan"))
            lr        = self._current_lr()

            # ── Epoch summary ────────────────────────────────────────────
            log.info(
                "Epoch %d/%d — train_loss=%.4f  val_pcc_total=%.4f  "
                "val_mse=%.4f  lr=%.2e  elapsed=%s",
                epoch, epochs, train_loss, pcc_total, mse_total,
                lr, self._elapsed_str(),
            )
            dim_line = "  ".join(
                f"{k.replace('val_pcc_','')[:4]}={val_metrics.get('val_pcc_'+k, float('nan')):.3f}"
                for k in FUSION_SCORE_KEYS
            )
            log.info("  PCC → %s", dim_line)

            # ── Val CSV ──────────────────────────────────────────────────
            self._val_csv.write({
                "epoch":                epoch,
                "step":                 self.global_step,
                "train_loss":           _fmt(train_loss),
                "val_pcc_total":        _fmt(val_metrics.get("val_pcc_total")),
                "val_pcc_accuracy":     _fmt(val_metrics.get("val_pcc_accuracy")),
                "val_pcc_fluency":      _fmt(val_metrics.get("val_pcc_fluency")),
                "val_pcc_prosodic":     _fmt(val_metrics.get("val_pcc_prosodic")),
                "val_pcc_completeness": _fmt(val_metrics.get("val_pcc_completeness")),
                "val_mse_total":        _fmt(mse_total),
                "lr":                   _fmt(lr, prec=8),
                "elapsed_sec":          self._elapsed(),
            })

            # ── Checkpoint management ────────────────────────────────────
            self._manage_checkpoints(epoch, val_metrics)

            # ── Early stopping ───────────────────────────────────────────
            if early_stop and self.patience_count >= patience:
                log.info(
                    "Early stopping: %s has not improved by %.4f for %d epochs.",
                    metric_key, tcfg.get("min_delta", 0.001), patience,
                )
                break

        self._step_csv.close()
        self._val_csv.close()
        if self.syncer:
            self.syncer.wait_all()
        log.info(
            "Training complete. Best %s=%.4f → %s",
            metric_key, self.best_metric,
            self.best_ckpt_path or "not saved",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Resume
    # ──────────────────────────────────────────────────────────────────────

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.global_step   = ckpt["global_step"]
        self.start_epoch   = ckpt["epoch"]
        self.best_metric   = ckpt.get("best_metric", float("-inf"))
        log.info(
            "Resumed from %s  (step=%d, epoch=%d, best=%.4f)",
            path, self.global_step, self.start_epoch, self.best_metric,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Run-dir setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_run_dir(cfg: dict, resume_run_dir: Optional[str] = None) -> Path:
    log_root = cfg.get("logging", {}).get("tb_log_dir", "logs/finetune")
    if resume_run_dir and Path(resume_run_dir).exists():
        run_dir = Path(resume_run_dir)
    else:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(log_root) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write config snapshot on fresh runs (not resume).
    snap = run_dir / "config_snapshot.yaml"
    if not snap.exists():
        with open(snap, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False)
    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion-C fine-tuning")
    parser.add_argument("--config", default="configs/finetune_config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint to resume from.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # Determine run directory (resume or fresh).
    resume_run_dir = None
    if args.resume:
        try:
            ckpt_meta = torch.load(args.resume, map_location="cpu")
            resume_run_dir = ckpt_meta.get("run_dir")
        except Exception:
            pass

    run_dir = _setup_run_dir(cfg, resume_run_dir)
    log.info("Run directory: %s", run_dir)

    # Data
    log.info("Building datasets …")
    dcfg = cfg["data"]
    train_ds = SpeechoceanFusionDataset(cfg, split=dcfg.get("train_split", "train"), augment=False)
    val_ds   = SpeechoceanFusionDataset(cfg, split=dcfg.get("test_split",  "test"),  augment=False)
    log.info("  train=%d  val=%d utterances", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size  = dcfg["batch_size"],
        shuffle     = True,
        num_workers = dcfg.get("num_workers", 4),
        collate_fn  = fusion_collate_fn,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = dcfg["batch_size"],
        shuffle     = False,
        num_workers = dcfg.get("num_workers", 4),
        collate_fn  = fusion_collate_fn,
        pin_memory  = torch.cuda.is_available(),
    )

    # Model
    log.info("Building model …")
    model = MultiResHuBERTFinetune(cfg)

    # Trainer
    syncer  = build_syncer(cfg)
    trainer = FusionCTrainer(model, train_loader, val_loader, cfg, run_dir, syncer)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
