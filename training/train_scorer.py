"""
Training script for the PronunciationScorer.

Entry point:
    python training/train_scorer.py --config configs/scoring_config.yaml
    python training/train_scorer.py --config configs/scoring_config.yaml --resume checkpoints/scorer/last.pt

Outputs per run:
    logs/scorer/run_YYYYMMDD_HHMMSS/
    ├── train_steps.csv     step-level: step, epoch, loss, loss_<dim>×5, lr, grad_norm, elapsed_sec
    ├── val_epochs.csv      epoch-level: epoch, val_pcc_<dim>×5, val_mse_total, lr, elapsed_sec
    └── config_snapshot.yaml

Checkpoints:
    checkpoints/scorer/
    ├── best_model.pt       best val_pcc_total
    └── last_epoch_NNN.pt   rolling last-N
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.speechocean_asr import SpeechOceanASRDataset, asr_collate_fn, SCORE_KEYS
from model.pronunciation_scorer import PronunciationScorer
from training.scheduler import get_warmup_cosine_schedule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DIM_NAMES: List[str] = SCORE_KEYS   # ["total", "accuracy", "fluency", "prosodic", "completeness"]


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def smoothed_mse(pred: Tensor, target: Tensor, alpha: float = 0.0) -> Tensor:
    """MSE with optional label smoothing toward the batch mean."""
    if alpha == 0.0:
        return F.mse_loss(pred, target)
    target_smooth = (1.0 - alpha) * target + alpha * target.mean()
    return F.mse_loss(pred, target_smooth)


def _pearson_r_1d(pred: Tensor, target: Tensor) -> Tensor:
    vp  = pred   - pred.mean()
    vt  = target - target.mean()
    return (vp * vt).sum() / (vp.norm() * vt.norm() + 1e-8)


def pcc_loss(pred: Tensor, target: Tensor) -> Tensor:
    """1 − Pearson r.  Accepts [B] or [B, D]."""
    if pred.dim() == 1:
        return 1.0 - _pearson_r_1d(pred, target)
    vp = pred   - pred.mean(0, keepdim=True)
    vt = target - target.mean(0, keepdim=True)
    r  = (vp * vt).sum(0) / (vp.norm(dim=0) * vt.norm(dim=0) + 1e-8)
    return (1.0 - r).mean()


def compute_loss(
    pred:         Tensor,
    target:       Tensor,
    mse_weight:   float,
    pcc_weight:   float,
    smooth_alpha: float = 0.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """Per-dimension (smoothed MSE + PCC) loss, averaged over 5 dims."""
    losses: List[Tensor]       = []
    dim_losses: Dict[str, float] = {}
    for i, name in enumerate(_DIM_NAMES):
        p     = pred[:, i]
        t     = target[:, i]
        d_loss = mse_weight * smoothed_mse(p, t, alpha=smooth_alpha) \
               + pcc_weight * pcc_loss(p, t)
        dim_losses[f"loss_{name}"] = d_loss.item()
        losses.append(d_loss)
    return torch.stack(losses).mean(), dim_losses


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

_STEP_COLS = (
    ["step", "epoch", "loss"]
    + [f"loss_{n}" for n in _DIM_NAMES]
    + ["lr", "grad_norm", "elapsed_sec"]
)
_EPOCH_COLS = (
    ["epoch", "step", "train_loss"]
    + [f"val_pcc_{n}" for n in _DIM_NAMES]
    + ["val_mse_total", "lr", "elapsed_sec"]
)


def _fmt(v: Optional[float], prec: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.{prec}f}"


class _CSVWriter:
    def __init__(self, path: Path, cols: List[str], append: bool = False) -> None:
        mode       = "a" if append else "w"
        self._f   = open(path, mode, newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._f, fieldnames=cols, extrasaction="ignore")
        if not append:
            self._csv.writeheader()
            self._f.flush()

    def write(self, row: dict) -> None:
        self._csv.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    cfg_file = Path(path).resolve()
    cfg_dir  = cfg_file.parent
    cfg      = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    for section, key in [
        ("model",    "pretrain_checkpoint"),
        ("training", "output_dir"),
        ("logging",  "tb_log_dir"),
        ("data",     "asr_cache_path"),
    ]:
        if section not in cfg:
            continue
        raw = (cfg[section] or {}).get(key)
        if raw and raw != "null" and not Path(raw).is_absolute():
            cfg[section][key] = str(cfg_dir / raw)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ScorerTrainer:
    """
    Training loop for PronunciationScorer with gradient accumulation,
    AMP, cosine LR scheduling, and per-dim CSV logging.
    """

    def __init__(
        self,
        model:        PronunciationScorer,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          dict,
        run_dir:      Path,
    ) -> None:
        tcfg = cfg["training"]

        self.cfg          = cfg
        self.tcfg         = tcfg
        self.device       = torch.device(
            tcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.run_dir      = run_dir
        self.output_dir   = Path(tcfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Gradient accumulation + label smoothing
        self.accum_steps  = tcfg.get("grad_accumulation_steps", 1)
        self.smooth_alpha = tcfg.get("label_smooth_alpha", 0.0)
        self.mse_w        = tcfg.get("mse_weight", 0.5)
        self.pcc_w        = tcfg.get("pcc_weight", 0.5)

        # Optimizer + scheduler (total_steps = optimizer steps, not micro-batches)
        self.optimizer    = model.get_optimizer()
        micro_per_epoch   = len(train_loader)
        steps_per_epoch   = max(1, micro_per_epoch // self.accum_steps)
        total_steps       = tcfg["epochs"] * steps_per_epoch
        self.scheduler    = get_warmup_cosine_schedule(
            self.optimizer,
            warmup_steps = tcfg.get("warmup_steps", 500),
            total_steps  = total_steps,
            min_lr_ratio = tcfg.get("min_lr", 1e-6) / max(
                tcfg["lr_speech_encoder"], tcfg["lr_audio_proj"], tcfg["lr_fusion"]
            ),
        )

        self.scaler = GradScaler(
            enabled = tcfg.get("use_amp", True) and self.device.type == "cuda"
        )

        # State
        self.best_metric   = float("-inf")
        self.patience_count = 0
        self.best_ckpt_path: Optional[Path] = None
        self._last_ckpts:    List[Path]     = []
        self.global_step   = 0
        self.start_epoch   = 0

        # CSV loggers
        self._step_csv = _CSVWriter(run_dir / "train_steps.csv",  _STEP_COLS)
        self._val_csv  = _CSVWriter(run_dir / "val_epochs.csv",   _EPOCH_COLS)
        self._t0       = time.monotonic()

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _elapsed_str(self) -> str:
        return str(timedelta(seconds=int(time.monotonic() - self._t0)))

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._t0, 1)

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    # ──────────────────────────────────────────────────────────────────────
    # Training epoch (gradient accumulation)
    # ──────────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        accum_steps = self.accum_steps
        log_every   = self.tcfg.get("log_every", 20)
        use_amp     = self.tcfg.get("use_amp", True) and self.device.type == "cuda"
        grad_clip   = self.tcfg.get("grad_clip", 1.0)

        total_loss   = 0.0
        n_micro      = 0
        window_loss  = 0.0
        window_micro = 0
        window_dims: Dict[str, float] = {}

        self.optimizer.zero_grad(set_to_none=True)

        for micro_idx, batch in enumerate(self.train_loader):
            waveforms   = batch["waveforms"].to(self.device)
            attn_mask   = batch["attention_mask"].to(self.device)
            targets     = batch["labels"].to(self.device)
            transcripts = batch["transcripts"]

            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                pred = self.model(waveforms, attn_mask, transcripts)
                loss, dim_losses = compute_loss(
                    pred, targets, self.mse_w, self.pcc_w, self.smooth_alpha
                )

            self.scaler.scale(loss / accum_steps).backward()

            total_loss   += loss.item()
            n_micro      += 1
            window_loss  += loss.item()
            window_micro += 1
            for k, v in dim_losses.items():
                window_dims[k] = window_dims.get(k, 0.0) + v

            is_boundary = (micro_idx + 1) % accum_steps == 0
            is_last     = (micro_idx + 1) == len(self.train_loader)

            if is_boundary or is_last:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), grad_clip
                ).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                step_loss = window_loss  / window_micro
                step_dims = {k: v / window_micro for k, v in window_dims.items()}

                if self.global_step % log_every == 0:
                    dim_str = "  ".join(
                        f"{name[:3]}={step_dims[f'loss_{name}']:.3f}"
                        for name in _DIM_NAMES
                    )
                    log.info(
                        "[Ep %d | Step %d] loss=%.4f  %s  lr=%.2e gnorm=%.2f | %s",
                        epoch, self.global_step, step_loss, dim_str,
                        self._current_lr(), grad_norm, self._elapsed_str(),
                    )
                    self._step_csv.write({
                        "step":        self.global_step,
                        "epoch":       epoch,
                        "loss":        _fmt(step_loss),
                        **{k: _fmt(step_dims[k]) for k in step_dims},
                        "lr":          _fmt(self._current_lr(), prec=8),
                        "grad_norm":   _fmt(grad_norm),
                        "elapsed_sec": self._elapsed(),
                    })

                window_loss  = 0.0
                window_micro = 0
                window_dims  = {}

        return total_loss / max(1, n_micro)

    # ──────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        all_pred:   List[np.ndarray] = []
        all_target: List[np.ndarray] = []

        for batch in self.val_loader:
            waveforms   = batch["waveforms"].to(self.device)
            attn_mask   = batch["attention_mask"].to(self.device)
            targets     = batch["labels"]
            transcripts = batch["transcripts"]
            pred        = self.model(waveforms, attn_mask, transcripts)
            all_pred.append(pred.cpu().float().numpy())
            all_target.append(targets.float().numpy())

        preds   = np.concatenate(all_pred,   axis=0)   # [N, 5]
        targets = np.concatenate(all_target, axis=0)   # [N, 5]

        metrics: Dict[str, float] = {}
        for i, key in enumerate(_DIM_NAMES):
            p, t  = preds[:, i], targets[:, i]
            corr  = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-6 else float("nan")
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
        return path

    def _manage_checkpoints(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        tcfg       = self.tcfg
        save_every = tcfg.get("save_every_n_epochs", 5)
        keep_n     = tcfg.get("keep_last_n_checkpoints", 3)
        metric_key = tcfg.get("best_metric", "val_pcc_total")
        min_delta  = tcfg.get("min_delta", 0.001)
        current    = val_metrics.get(metric_key, float("-inf"))

        if current - self.best_metric > min_delta:
            self.best_metric    = current
            self.patience_count = 0
            self.best_ckpt_path = self._save("best_model.pt", epoch, val_metrics)
            log.info("  * New best %s=%.4f → best_model.pt", metric_key, current)
        else:
            self.patience_count += 1

        if epoch % save_every == 0:
            name = f"last_epoch_{epoch:03d}.pt"
            p    = self._save(name, epoch, val_metrics)
            self._last_ckpts.append(p)
            while len(self._last_ckpts) > keep_n:
                old = self._last_ckpts.pop(0)
                if old.exists() and old != self.best_ckpt_path:
                    old.unlink()

    # ──────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        tcfg       = self.tcfg
        epochs     = tcfg["epochs"]
        patience   = tcfg.get("patience", 10)
        early_stop = tcfg.get("early_stopping", True)
        metric_key = tcfg.get("best_metric", "val_pcc_total")

        log.info(
            "Training PronunciationScorer | device=%s | epochs=%d | "
            "lr_speech=%.1e | lr_proj=%.1e | lr_fusion=%.1e",
            self.device, epochs,
            tcfg["lr_speech_encoder"], tcfg["lr_audio_proj"], tcfg["lr_fusion"],
        )

        for epoch in range(self.start_epoch + 1, epochs + 1):
            log.info("── Epoch %d / %d ──", epoch, epochs)
            train_loss  = self._train_epoch(epoch)
            val_metrics = self._validate()

            pcc_total = val_metrics.get("val_pcc_total", float("nan"))
            mse_total = val_metrics.get("val_mse_total", float("nan"))
            lr        = self._current_lr()

            log.info(
                "Epoch %d/%d — train=%.4f  val_pcc_total=%.4f  val_mse=%.4f  lr=%.2e | %s",
                epoch, epochs, train_loss, pcc_total, mse_total, lr, self._elapsed_str(),
            )
            dim_line = "  ".join(
                f"{k[:3]}={val_metrics.get(f'val_pcc_{k}', float('nan')):.3f}"
                for k in _DIM_NAMES
            )
            log.info("  PCC → %s", dim_line)

            self._val_csv.write({
                "epoch":        epoch,
                "step":         self.global_step,
                "train_loss":   _fmt(train_loss),
                **{f"val_pcc_{k}": _fmt(val_metrics.get(f"val_pcc_{k}")) for k in _DIM_NAMES},
                "val_mse_total": _fmt(mse_total),
                "lr":           _fmt(lr, prec=8),
                "elapsed_sec":  self._elapsed(),
            })

            self._manage_checkpoints(epoch, val_metrics)

            if early_stop and self.patience_count >= patience:
                log.info(
                    "Early stopping: %s not improved for %d epochs.", metric_key, patience
                )
                break

        self._step_csv.close()
        self._val_csv.close()
        log.info(
            "Training complete. Best %s=%.4f → %s",
            metric_key, self.best_metric,
            self.best_ckpt_path or "not saved",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint resume
    # ──────────────────────────────────────────────────────────────────────

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        except (ValueError, RuntimeError) as exc:
            log.warning(
                "Optimizer state mismatch (%s). Restoring per-group LRs only.", exc
            )
            saved = ckpt["optimizer_state"].get("param_groups", [])
            for i, pg in enumerate(self.optimizer.param_groups):
                if i < len(saved):
                    pg["lr"] = saved[i]["lr"]
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.global_step  = ckpt["global_step"]
        self.start_epoch  = ckpt["epoch"]
        self.best_metric  = ckpt.get("best_metric", float("-inf"))
        log.info(
            "Resumed from %s  (step=%d, epoch=%d, best=%.4f)",
            path, self.global_step, self.start_epoch, self.best_metric,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Run directory setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_run_dir(cfg: dict, resume_run_dir: Optional[str] = None) -> Path:
    log_root = cfg.get("logging", {}).get("tb_log_dir", "logs/scorer")
    if resume_run_dir and Path(resume_run_dir).exists():
        run_dir = Path(resume_run_dir)
    else:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(log_root) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    snap = run_dir / "config_snapshot.yaml"
    if not snap.exists():
        snap.write_text(
            yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PronunciationScorer")
    parser.add_argument("--config", default="configs/scoring_config.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    resume_run_dir = None
    if args.resume:
        try:
            meta = torch.load(args.resume, map_location="cpu")
            resume_run_dir = meta.get("run_dir")
        except Exception:
            pass

    run_dir = _setup_run_dir(cfg, resume_run_dir)
    log.info("Run directory: %s", run_dir)

    dcfg = cfg["data"]
    log.info("Building datasets …")
    train_ds = SpeechOceanASRDataset(cfg, split=dcfg.get("train_split", "train"))
    val_ds   = SpeechOceanASRDataset(cfg, split=dcfg.get("test_split",  "test"))
    log.info("  train=%d  val=%d utterances", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size  = dcfg["batch_size"],
        shuffle     = True,
        num_workers = dcfg.get("num_workers", 4),
        collate_fn  = asr_collate_fn,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = dcfg["batch_size"],
        shuffle     = False,
        num_workers = dcfg.get("num_workers", 4),
        collate_fn  = asr_collate_fn,
        pin_memory  = torch.cuda.is_available(),
    )

    log.info("Building model …")
    model = PronunciationScorer(cfg)

    trainer = ScorerTrainer(model, train_loader, val_loader, cfg, run_dir)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
