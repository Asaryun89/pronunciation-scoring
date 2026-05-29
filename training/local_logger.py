"""
Local run logger for Multi-resolution HuBERT training.

Creates a timestamped run directory and writes:
  logs/run_YYYYMMDD_HHMMSS/
  ├── train.log       full text log (every step and epoch)
  ├── metrics.csv     epoch-level metrics for plotting
  └── config.yaml     copy of the YAML config used for this run

Console output uses rich if available, otherwise plain stdlib logging.
Resume support: pass resume_dir to append to an existing run directory.
"""

from __future__ import annotations

import csv
import logging
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

log = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.logging import RichHandler
    _RICH = True
except ImportError:
    _RICH = False


# ─────────────────────────────────────────────────────────────────────────────
# CSV column sets per training mode
# ─────────────────────────────────────────────────────────────────────────────

_PRETRAIN_COLS = [
    "epoch", "step",
    "train_loss", "train_loss_hi", "train_loss_lo",
    "train_acc_hi", "train_acc_lo",
    "lr", "grad_norm", "elapsed_sec",
]

_FINETUNE_COLS = [
    "epoch", "step",
    "train_loss", "val_loss",
    "val_pcc_total", "val_pcc_accuracy", "val_pcc_fluency",
    "val_pcc_prosodic", "val_pcc_completeness",
    "val_mse", "lr", "elapsed_sec",
]


# ─────────────────────────────────────────────────────────────────────────────

class LocalRunLogger:
    """
    Local-only run logger.  All output stays on disk — no cloud services.

    Args:
        log_root:   Parent directory for run folders (e.g. ``"logs"``).
        mode:       ``"pretrain"`` or ``"finetune"`` — determines CSV columns.
        config:     Full config dict written to ``config.yaml`` (first run only).
        resume_dir: Existing run directory to append to when resuming.
                    If None or the path doesn't exist, a new run dir is created.
    """

    def __init__(
        self,
        log_root:   str,
        mode:       str            = "pretrain",
        config:     Optional[Dict] = None,
        resume_dir: Optional[str]  = None,
    ) -> None:
        self._mode  = mode
        self._start = time.monotonic()
        resuming    = False

        if resume_dir and Path(resume_dir).exists():
            self.run_dir = Path(resume_dir)
            resuming = True
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(log_root) / f"run_{ts}"

        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self._setup_handlers(resuming)

        if config and not resuming:
            self._write_config(config)

        cols = _PRETRAIN_COLS if mode == "pretrain" else _FINETUNE_COLS
        self._setup_csv(cols, append=resuming)

        if resuming:
            sep = "=" * 60
            resume_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._file_handler.stream.write(
                f"\n{sep}\n--- RESUMED  {resume_ts} ---\n{sep}\n\n"
            )
            self._file_handler.stream.flush()
            log.info("Appending to existing run: %s", self.run_dir)
        else:
            log.info("New run directory: %s", self.run_dir)

    # ──────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────

    def _setup_handlers(self, resuming: bool) -> None:
        log_path = self.run_dir / "train.log"
        mode = "a" if resuming else "w"

        root = logging.getLogger()

        # Replace any existing plain StreamHandlers (set by basicConfig) with
        # a rich-formatted one so we don't get duplicate console output.
        for h in list(root.handlers):
            if type(h) is logging.StreamHandler:
                root.removeHandler(h)
                h.close()

        # ── console handler ───────────────────────────────────────────────
        if _RICH:
            _con = Console(highlight=False)
            ch = RichHandler(console=_con, show_path=False, markup=True,
                             rich_tracebacks=True, log_time_format="[%H:%M:%S]")
        else:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            ))
        ch.setLevel(logging.INFO)
        root.addHandler(ch)

        # ── file handler (plain text, append-safe) ────────────────────────
        fh = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(fh)
        self._file_handler = fh

    def _write_config(self, config: Dict) -> None:
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False)

    def _setup_csv(self, cols: list, append: bool) -> None:
        mode = "a" if append else "w"
        self._csv_f = open(self.run_dir / "metrics.csv", mode,
                           newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_f, fieldnames=cols,
                                   extrasaction="ignore")
        if not append:
            self._csv.writeheader()
            self._csv_f.flush()

    # ──────────────────────────────────────────────────────────────────────
    # Elapsed helpers
    # ──────────────────────────────────────────────────────────────────────

    def _elapsed_str(self) -> str:
        return str(timedelta(seconds=int(time.monotonic() - self._start)))

    def _elapsed_sec(self) -> float:
        return round(time.monotonic() - self._start, 1)

    # ──────────────────────────────────────────────────────────────────────
    # Public logging interface
    # ──────────────────────────────────────────────────────────────────────

    def log_step(
        self,
        step:          int,
        epoch:         int,
        total_epochs:  int,
        metrics:       Dict[str, float],
        tokens_per_sec: Optional[float] = None,
    ) -> None:
        """
        Write a single step summary line to the console and train.log.
        Called every ``log_every`` optimizer steps.
        """
        elapsed = self._elapsed_str()
        hdr = f"[Epoch {epoch}/{total_epochs} | Step {step}]"

        if self._mode == "pretrain":
            loss    = metrics.get("train/loss",      float("nan"))
            loss_hi = metrics.get("train/loss_hi",   float("nan"))
            loss_lo = metrics.get("train/loss_lo",   float("nan"))
            acc_hi  = metrics.get("train/acc_hi",    float("nan"))
            lr      = metrics.get("train/lr",        float("nan"))
            gnorm   = metrics.get("train/grad_norm", float("nan"))
            body = (
                f"loss={loss:.4f} (hi={loss_hi:.4f} lo={loss_lo:.4f}) | "
                f"acc_hi={acc_hi:.3f} | lr={lr:.2e} | gnorm={gnorm:.2f}"
            )
        else:
            loss = metrics.get("train/loss", float("nan"))
            lr   = metrics.get("train/lr",   float("nan"))
            body = f"loss={loss:.4f} | lr={lr:.2e}"

        tps = f" | {tokens_per_sec:.1f} tok/s" if tokens_per_sec is not None else ""
        log.info("%s %s%s | %s elapsed", hdr, body, tps, elapsed)

    def log_epoch(
        self,
        epoch:        int,
        total_epochs: int,
        step:         int,
        metrics:      Dict[str, float],
    ) -> None:
        """
        Write an epoch summary line to train.log + console and one CSV row.
        Called at the end of every epoch.
        """
        elapsed     = self._elapsed_str()
        elapsed_sec = self._elapsed_sec()
        hdr         = f"[Epoch {epoch}/{total_epochs} | Step {step}]"

        train_loss = metrics.get("epoch/train_loss",
                     metrics.get("train/loss", float("nan")))
        lr         = metrics.get("train/lr", float("nan"))

        if self._mode == "finetune":
            val_loss  = metrics.get("epoch/val_loss",  float("nan"))
            pcc_total = metrics.get("val/pcc_total",   float("nan"))
            mse_total = metrics.get("val/mse_total",   float("nan"))
            log.info(
                "%s loss=%s | val_loss=%s | pcc=%s | mse=%s | lr=%s | %s elapsed",
                hdr,
                f"{train_loss:.4f}", f"{val_loss:.4f}",
                f"{pcc_total:.4f}",  f"{mse_total:.4f}",
                f"{lr:.2e}",         elapsed,
            )
            row: Dict[str, Any] = {
                "epoch":                epoch,
                "step":                 step,
                "train_loss":           _fmt(train_loss),
                "val_loss":             _fmt(val_loss),
                "val_pcc_total":        _fmt(metrics.get("val/pcc_total")),
                "val_pcc_accuracy":     _fmt(metrics.get("val/pcc_accuracy")),
                "val_pcc_fluency":      _fmt(metrics.get("val/pcc_fluency")),
                "val_pcc_prosodic":     _fmt(metrics.get("val/pcc_prosodic")),
                "val_pcc_completeness": _fmt(metrics.get("val/pcc_completeness")),
                "val_mse":              _fmt(metrics.get("val/mse_total")),
                "lr":                   _fmt(lr, precision=6),
                "elapsed_sec":          elapsed_sec,
            }
        else:
            acc_hi = metrics.get("train/acc_hi",    float("nan"))
            acc_lo = metrics.get("train/acc_lo",    float("nan"))
            gnorm  = metrics.get("train/grad_norm", float("nan"))
            log.info(
                "%s loss=%s | acc_hi=%s acc_lo=%s | lr=%s | %s elapsed",
                hdr,
                f"{train_loss:.4f}",
                f"{acc_hi:.3f}", f"{acc_lo:.3f}",
                f"{lr:.2e}",     elapsed,
            )
            row = {
                "epoch":         epoch,
                "step":          step,
                "train_loss":    _fmt(train_loss),
                "train_loss_hi": _fmt(metrics.get("train/loss_hi")),
                "train_loss_lo": _fmt(metrics.get("train/loss_lo")),
                "train_acc_hi":  _fmt(acc_hi),
                "train_acc_lo":  _fmt(acc_lo),
                "lr":            _fmt(lr, precision=6),
                "grad_norm":     _fmt(gnorm),
                "elapsed_sec":   elapsed_sec,
            }

        self._csv.writerow(row)
        self._csv_f.flush()

    def info(self, msg: str, *args: Any) -> None:
        """Log a plain info message (written to both console and train.log)."""
        log.info(msg, *args)

    def close(self) -> None:
        """Flush and close all file handles."""
        try:
            self._csv_f.close()
        except Exception:
            pass
        try:
            root = logging.getLogger()
            root.removeHandler(self._file_handler)
            self._file_handler.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: Optional[float], precision: int = 4) -> str:
    """Format a float for CSV; 'nan' for missing/NaN values."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.{precision}f}"
