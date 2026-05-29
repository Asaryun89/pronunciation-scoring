#!/usr/bin/env python3
"""
Plot training metrics from a LocalRunLogger run directory.

Reads ``metrics.csv`` and saves figures to ``<run_dir>/plots/``.

Usage
─────
    # Pretrain run
    python plot_results.py logs/pretrain/run_20240101_120000

    # Fine-tune run
    python plot_results.py logs/finetune/run_20240101_130000

    # Overlay multiple runs for comparison
    python plot_results.py logs/finetune/run_A logs/finetune/run_B --labels "exp-A" "exp-B"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _read_csv(run_dir: Path) -> List[Dict[str, str]]:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        print(f"[ERROR] metrics.csv not found in {run_dir}", file=sys.stderr)
        sys.exit(1)
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _col(rows: List[Dict[str, str]], key: str) -> List[float]:
    """Extract a numeric column, skipping nan/missing values (returns nan for those)."""
    import math
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            out.append(float("nan"))
    return out


def _has_col(rows: List[Dict[str, str]], key: str) -> bool:
    return bool(rows) and key in rows[0]


def _detect_mode(rows: List[Dict[str, str]]) -> str:
    if _has_col(rows, "val_loss"):
        return "finetune"
    return "pretrain"


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")


def plot_pretrain(
    all_rows:   List[List[Dict[str, str]]],
    labels:     List[str],
    output_dir: Path,
) -> None:
    import math
    import matplotlib.pyplot as plt

    epochs_list = [_col(r, "epoch")         for r in all_rows]
    loss_list   = [_col(r, "train_loss")    for r in all_rows]
    hi_list     = [_col(r, "train_loss_hi") for r in all_rows]
    lo_list     = [_col(r, "train_loss_lo") for r in all_rows]
    acc_hi_list = [_col(r, "train_acc_hi")  for r in all_rows]
    acc_lo_list = [_col(r, "train_acc_lo")  for r in all_rows]
    lr_list     = [_col(r, "lr")            for r in all_rows]
    gnorm_list  = [_col(r, "grad_norm")     for r in all_rows]

    # ── Loss ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for ep, loss, lbl in zip(epochs_list, loss_list, labels):
        ax.plot(ep, loss, label=lbl)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Pre-training loss"); ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, output_dir / "loss.png"); plt.close(fig)

    # ── Per-head loss ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ep, hi, lo, lbl in zip(epochs_list, hi_list, lo_list, labels):
        axes[0].plot(ep, hi, label=lbl)
        axes[1].plot(ep, lo, label=lbl)
    for ax, title in zip(axes, ["Loss hi (H₃)", "Loss lo (H₂)"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir / "loss_per_head.png"); plt.close(fig)

    # ── Accuracy ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ep, hi, lo, lbl in zip(epochs_list, acc_hi_list, acc_lo_list, labels):
        axes[0].plot(ep, hi, label=lbl)
        axes[1].plot(ep, lo, label=lbl)
    for ax, title in zip(axes, ["Accuracy hi (H₃)", "Accuracy lo (H₂)"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir / "accuracy.png"); plt.close(fig)

    # ── LR & grad norm ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ep, lr, gn, lbl in zip(epochs_list, lr_list, gnorm_list, labels):
        axes[0].plot(ep, lr, label=lbl)
        axes[1].plot(ep, gn, label=lbl)
    axes[0].set_title("Learning rate"); axes[0].set_yscale("log")
    axes[1].set_title("Gradient norm")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir / "lr_and_gradnorm.png"); plt.close(fig)


def plot_finetune(
    all_rows:   List[List[Dict[str, str]]],
    labels:     List[str],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    PCC_DIMS = ["accuracy", "fluency", "completeness", "prosodic", "total"]
    epochs_list    = [_col(r, "epoch")        for r in all_rows]
    train_loss_list= [_col(r, "train_loss")   for r in all_rows]
    val_loss_list  = [_col(r, "val_loss")     for r in all_rows]
    lr_list        = [_col(r, "lr")           for r in all_rows]

    # ── Train / val loss ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for ep, tl, vl, lbl in zip(epochs_list, train_loss_list, val_loss_list, labels):
        ax.plot(ep, tl, label=f"{lbl} train", linestyle="--")
        ax.plot(ep, vl, label=f"{lbl} val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Fine-tuning loss"); ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, output_dir / "loss.png"); plt.close(fig)

    # ── PCC per dimension ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(PCC_DIMS), figsize=(4 * len(PCC_DIMS), 4))
    for dim, ax in zip(PCC_DIMS, axes):
        col = f"val_pcc_{dim}"
        for ep, rows, lbl in zip(epochs_list, all_rows, labels):
            if _has_col(rows, col):
                ax.plot(ep, _col(rows, col), label=lbl)
        ax.set_title(f"PCC {dim}"); ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir / "pcc_per_dim.png"); plt.close(fig)

    # ── MSE ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for ep, rows, lbl in zip(epochs_list, all_rows, labels):
        col = "val_mse"
        if _has_col(rows, col):
            ax.plot(ep, _col(rows, col), label=lbl)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title("Validation MSE (total)"); ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, output_dir / "mse.png"); plt.close(fig)

    # ── Learning rate ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for ep, lr, lbl in zip(epochs_list, lr_list, labels):
        ax.plot(ep, lr, label=lbl)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_yscale("log")
    ax.set_title("Learning rate"); ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, output_dir / "lr.png"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LocalRunLogger metrics")
    parser.add_argument("run_dirs", nargs="+", help="One or more run directories")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Legend labels (one per run dir, defaults to dir name)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save plots (default: <first_run_dir>/plots/)")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    labels   = args.labels or [d.name for d in run_dirs]

    if len(labels) != len(run_dirs):
        parser.error(f"--labels count ({len(labels)}) must match run_dirs count ({len(run_dirs)})")

    all_rows = [_read_csv(d) for d in run_dirs]
    mode     = _detect_mode(all_rows[0])
    out_dir  = Path(args.output_dir) if args.output_dir else run_dirs[0] / "plots"

    print(f"Mode: {mode}  |  runs: {len(run_dirs)}  |  output: {out_dir}")

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("[ERROR] matplotlib is required:  pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    if mode == "pretrain":
        plot_pretrain(all_rows, labels, out_dir)
    else:
        plot_finetune(all_rows, labels, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
