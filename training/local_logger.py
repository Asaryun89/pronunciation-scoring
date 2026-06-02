"""
LocalRunLogger — CSV-based run logging for pre-training.

Creates a timestamped run directory and writes:
    train_steps.csv   step-level metrics
    train_epochs.csv  epoch-level averages
    config.yaml       snapshot of the config used
"""

from __future__ import annotations

import csv
import logging
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

log = logging.getLogger(__name__)

_STEP_COLS = [
    "step", "epoch", "total_epochs",
    "loss", "acc_hi", "acc_lo", "lr", "grad_norm",
    "tokens_per_sec", "elapsed",
]
_EPOCH_COLS = [
    "epoch", "total_epochs", "step",
    "loss", "acc_hi", "acc_lo", "elapsed",
]


def _fmt(v: Any, prec: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


class _CSV:
    def __init__(self, path: Path, cols: List[str], append: bool = False) -> None:
        mode      = "a" if append else "w"
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


class LocalRunLogger:
    """
    Creates a run directory and writes CSV logs for pre-training.

    Args:
        log_root:   Parent directory for run folders.
        mode:       Label prefix (e.g. "pretrain").
        config:     Full config dict — saved as config.yaml in the run dir.
        resume_dir: Existing run directory to append logs to (resume mode).
    """

    def __init__(
        self,
        log_root:   str,
        mode:       str = "pretrain",
        config:     Optional[Dict] = None,
        resume_dir: Optional[str] = None,
    ) -> None:
        self._t0 = time.monotonic()

        if resume_dir and Path(resume_dir).exists():
            self.run_dir = Path(resume_dir)
            append = True
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(log_root) / f"{mode}_{ts}"
            append = False

        self.run_dir.mkdir(parents=True, exist_ok=True)

        if config and not append:
            cfg_snap = self.run_dir / "config.yaml"
            cfg_snap.write_text(
                yaml.dump(config, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )

        self._step_csv  = _CSV(self.run_dir / "train_steps.csv",  _STEP_COLS,  append)
        self._epoch_csv = _CSV(self.run_dir / "train_epochs.csv", _EPOCH_COLS, append)
        log.info("LocalRunLogger → %s  (append=%s)", self.run_dir, append)

    def _elapsed(self) -> str:
        return str(timedelta(seconds=int(time.monotonic() - self._t0)))

    def log_step(
        self,
        step:           int,
        epoch:          int,
        total_epochs:   int,
        metrics:        Dict[str, float],
        tokens_per_sec: Optional[float] = None,
    ) -> None:
        self._step_csv.write({
            "step":          step,
            "epoch":         epoch,
            "total_epochs":  total_epochs,
            "loss":          _fmt(metrics.get("train/loss")),
            "acc_hi":        _fmt(metrics.get("train/acc_hi")),
            "acc_lo":        _fmt(metrics.get("train/acc_lo")),
            "lr":            _fmt(metrics.get("train/lr"), prec=8),
            "grad_norm":     _fmt(metrics.get("train/grad_norm")),
            "tokens_per_sec":_fmt(tokens_per_sec, prec=1),
            "elapsed":       self._elapsed(),
        })

    def log_epoch(
        self,
        epoch:        int,
        total_epochs: int,
        step:         int,
        metrics:      Dict[str, float],
    ) -> None:
        self._epoch_csv.write({
            "epoch":        epoch,
            "total_epochs": total_epochs,
            "step":         step,
            "loss":         _fmt(metrics.get("train/loss")),
            "acc_hi":       _fmt(metrics.get("train/acc_hi")),
            "acc_lo":       _fmt(metrics.get("train/acc_lo")),
            "elapsed":      self._elapsed(),
        })

    def close(self) -> None:
        self._step_csv.close()
        self._epoch_csv.close()
