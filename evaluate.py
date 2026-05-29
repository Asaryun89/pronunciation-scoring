#!/usr/bin/env python3
"""
Evaluate a fine-tuned Multi-resolution HuBERT on the Speechocean762 test split.

Outputs
───────
  1. Per-dimension PCC and MSE / RMSE table (console + optional CSV)
  2. Head-contribution analysis — hi-res (H₃) vs lo-res (H₂) ablation
  3. Optional scatter-plot grid (requires matplotlib)
  4. Predictions CSV for downstream analysis

Usage
─────
    python evaluate.py \\
        --config      configs/finetune.yaml \\
        --checkpoint  checkpoints/finetune/best_model.pt \\
        --output-dir  results/eval/

    # Enable ablation study and plots
    python evaluate.py ... --ablation --plot
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data import Speechocean762Dataset, collate_fn
from model import MultiResHuBERT
from model.multi_res_hubert import _mean_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCORE_KEYS   = ["accuracy", "fluency", "completeness", "prosodic", "total"]
SCORE_SCALE  = 10.0   # labels are stored normalised [0,1]; multiply for [0,10]


# ─────────────────────────────────────────────────────────────────────────────
# Table formatting
# ─────────────────────────────────────────────────────────────────────────────

def _sep(widths: List[int], char: str = "─") -> str:
    return "┼".join(char * (w + 2) for w in widths)


def _row(cells: List[str], widths: List[int]) -> str:
    return "│".join(f" {c:<{w}} " for c, w in zip(cells, widths))


def print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    widths = [max(len(h), max(len(r[i]) for r in rows))
              for i, h in enumerate(headers)]
    bar = _sep(widths)
    print(f"\n{title}")
    print(bar)
    print(_row(headers, widths))
    print(bar)
    for r in rows:
        print(_row(r, widths))
    print(bar)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg: dict, ckpt_path: str, device: torch.device) -> MultiResHuBERT:
    model = MultiResHuBERT(**cfg["model"])
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    log.info("Loaded checkpoint: %s", ckpt_path)
    return model


def build_test_loader(cfg: dict) -> DataLoader:
    ds = Speechocean762Dataset(
        data_dir    = cfg["data"]["val_dir"],
        scores_path = cfg["data"]["scores_path"],
        max_duration_s = cfg["data"].get("max_duration_s", 20.0),
        augment     = False,
    )
    log.info("Test set: %d utterances", len(ds))
    return DataLoader(
        ds,
        batch_size  = cfg["data"].get("batch_size", 16),
        shuffle     = False,
        num_workers = cfg["data"].get("num_workers", 4),
        collate_fn  = collate_fn,
        pin_memory  = False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:  MultiResHuBERT,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray | List[str]]:
    """
    Full forward pass over the test set.

    Returns a dict with:
      preds       (N, 5)  float32 — predicted scores in [0, 10]
      labels      (N, 5)  float32 — ground-truth MOS in [0, 10]
      h3_pooled   (N, H)  float32 — mean-pooled H₃ (high-res encoder output)
      h2_pooled   (N, H)  float32 — mean-pooled H₂ (low-res encoder output)
      feat_mask_hi (N, T_feat)  bool tensors (stored as list, variable length)
      feat_mask_lo (N, T_feat') bool tensors
      utt_ids     List[str]
    """
    all_preds:    List[Tensor] = []
    all_labels:   List[Tensor] = []
    all_h3:       List[Tensor] = []
    all_h2:       List[Tensor] = []
    all_mask_hi:  List[Tensor] = []
    all_mask_lo:  List[Tensor] = []
    all_utt_ids:  List[str]    = []

    for batch in loader:
        waveforms = batch["waveforms"].to(device)
        masks     = batch["attention_mask"].to(device)
        labels    = batch["labels"]

        out = model(waveforms, masks, apply_mask=False)

        # Reconstruct feature-level masks from model internals
        feat_len = out.h3.shape[1]
        feat_mask_hi = model._audio_mask_to_feat_mask(masks, feat_len)  # (B, T_feat)
        feat_mask_lo = feat_len and model._audio_mask_to_feat_mask(
            masks, out.h2.shape[1]
        )

        all_preds.append(out.scores.cpu() * SCORE_SCALE)
        all_labels.append(labels * SCORE_SCALE)
        all_h3.append(
            _mean_pool(out.h3, feat_mask_hi).cpu()
        )
        all_h2.append(
            _mean_pool(out.h2, feat_mask_lo).cpu()
        )
        all_mask_hi.append(feat_mask_hi.cpu())
        all_mask_lo.append(feat_mask_lo.cpu())
        all_utt_ids.extend(batch["utt_ids"])

    return {
        "preds":       torch.cat(all_preds,  0).numpy(),
        "labels":      torch.cat(all_labels, 0).numpy(),
        "h3_pooled":   torch.cat(all_h3,     0).numpy(),
        "h2_pooled":   torch.cat(all_h2,     0).numpy(),
        "utt_ids":     all_utt_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Head-contribution ablation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ablated_forward(
    model:        MultiResHuBERT,
    loader:       DataLoader,
    device:       torch.device,
    ablate:       str,   # "none" | "hi" | "lo"
) -> np.ndarray:
    """
    Run inference with one score-head branch zeroed out.

    ablate="hi"  → score head sees only the lo-res (H₂) representation
    ablate="lo"  → score head sees only the hi-res (H₃) representation
    ablate="none"→ full combined model (same as run_inference)

    Returns: (N, 5) numpy array of predicted scores in [0, 10].
    """
    head = model.score_head
    all_preds: List[Tensor] = []

    for batch in loader:
        waveforms = batch["waveforms"].to(device)
        masks     = batch["attention_mask"].to(device)

        out = model(waveforms, masks, apply_mask=False)

        feat_len    = out.h3.shape[1]
        feat_mask_hi = model._audio_mask_to_feat_mask(masks, feat_len)
        feat_mask_lo = model._audio_mask_to_feat_mask(masks, out.h2.shape[1])

        # Pool both branches
        hi_pooled = _mean_pool(out.h3, feat_mask_hi)   # (B, H)
        lo_pooled = _mean_pool(out.h2, feat_mask_lo)   # (B, H)

        hi_proj = head.hi_proj(hi_pooled)   # (B, D/2)
        lo_proj = head.lo_proj(lo_pooled)   # (B, D/2)

        if ablate == "hi":
            hi_proj = torch.zeros_like(hi_proj)
        elif ablate == "lo":
            lo_proj = torch.zeros_like(lo_proj)

        fused  = head.mlp(torch.cat([hi_proj, lo_proj], dim=-1))
        scores = torch.cat([h(fused) for h in head.score_heads], dim=-1)
        scores = torch.sigmoid(scores) * SCORE_SCALE
        all_preds.append(scores.cpu())

    return torch.cat(all_preds, 0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    preds:  np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute PCC, MSE, RMSE for each score dimension and their averages.

    Args:
        preds:  (N, 5) predicted scores in [0, 10]
        labels: (N, 5) ground-truth  in [0, 10]

    Returns:
        Flat dict with keys like pcc_total, mse_accuracy, rmse_fluency, …
    """
    metrics: Dict[str, float] = {}
    for i, key in enumerate(SCORE_KEYS):
        r, _  = pearsonr(preds[:, i], labels[:, i])
        mse   = float(np.mean((preds[:, i] - labels[:, i]) ** 2))
        rmse  = float(np.sqrt(mse))
        metrics[f"pcc_{key}"]  = float(r)
        metrics[f"mse_{key}"]  = mse
        metrics[f"rmse_{key}"] = rmse

    # Macro-averages across all 5 dimensions
    metrics["pcc_avg"]  = float(np.mean([metrics[f"pcc_{k}"]  for k in SCORE_KEYS]))
    metrics["mse_avg"]  = float(np.mean([metrics[f"mse_{k}"]  for k in SCORE_KEYS]))
    metrics["rmse_avg"] = float(np.mean([metrics[f"rmse_{k}"] for k in SCORE_KEYS]))
    return metrics


def print_main_table(metrics: Dict[str, float]) -> None:
    headers = ["Dimension", "PCC ↑", "MSE ↓", "RMSE ↓"]
    rows: List[List[str]] = []
    for key in SCORE_KEYS:
        rows.append([
            key,
            f"{metrics[f'pcc_{key}']:+.4f}",
            f"{metrics[f'mse_{key}']:.4f}",
            f"{metrics[f'rmse_{key}']:.4f}",
        ])
    rows.append([
        "── avg ──",
        f"{metrics['pcc_avg']:+.4f}",
        f"{metrics['mse_avg']:.4f}",
        f"{metrics['rmse_avg']:.4f}",
    ])
    print_table("Main Results (scores in [0, 10])", headers, rows)


def print_ablation_table(
    metrics_combined: Dict[str, float],
    metrics_hi_only:  Dict[str, float],
    metrics_lo_only:  Dict[str, float],
) -> None:
    headers = ["Head", "PCC(total)", "MSE(total)", "PCC(avg)", "MSE(avg)"]
    rows: List[List[str]] = [
        [
            "Combined (H₃+H₂)",
            f"{metrics_combined['pcc_total']:+.4f}",
            f"{metrics_combined['mse_total']:.4f}",
            f"{metrics_combined['pcc_avg']:+.4f}",
            f"{metrics_combined['mse_avg']:.4f}",
        ],
        [
            "Hi-res only (H₃)",
            f"{metrics_hi_only['pcc_total']:+.4f}",
            f"{metrics_hi_only['mse_total']:.4f}",
            f"{metrics_hi_only['pcc_avg']:+.4f}",
            f"{metrics_hi_only['mse_avg']:.4f}",
        ],
        [
            "Lo-res only (H₂)",
            f"{metrics_lo_only['pcc_total']:+.4f}",
            f"{metrics_lo_only['mse_total']:.4f}",
            f"{metrics_lo_only['pcc_avg']:+.4f}",
            f"{metrics_lo_only['mse_avg']:.4f}",
        ],
    ]
    print_table("Head Contribution Ablation (total + avg across dims)", headers, rows)


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions_csv(
    preds:   np.ndarray,
    labels:  np.ndarray,
    utt_ids: List[str],
    path:    Path,
) -> None:
    """Save per-utterance predictions and labels to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["utt_id"]
            + [f"pred_{k}" for k in SCORE_KEYS]
            + [f"true_{k}" for k in SCORE_KEYS]
        )
        for uid, p, t in zip(utt_ids, preds, labels):
            writer.writerow([uid] + list(p.round(4)) + list(t.round(4)))
    log.info("Predictions saved → %s", path)


def save_metrics_csv(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in sorted(metrics.items()):
            writer.writerow([k, f"{v:.6f}"])
    log.info("Metrics saved → %s", path)


def plot_scatter(
    preds:   np.ndarray,
    labels:  np.ndarray,
    metrics: Dict[str, float],
    out_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping scatter plots.")
        return

    fig, axes = plt.subplots(1, len(SCORE_KEYS), figsize=(5 * len(SCORE_KEYS), 4))
    for i, (ax, key) in enumerate(zip(axes, SCORE_KEYS)):
        ax.scatter(labels[:, i], preds[:, i], alpha=0.35, s=10, color="steelblue")
        ax.plot([0, 10], [0, 10], "r--", linewidth=1, label="ideal")
        ax.set_xlabel(f"True {key}")
        ax.set_ylabel(f"Predicted {key}")
        pcc = metrics[f"pcc_{key}"]
        mse = metrics[f"mse_{key}"]
        ax.set_title(f"{key}\nPCC={pcc:.3f}  MSE={mse:.3f}")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    plt.suptitle("Predicted vs True MOS scores", fontsize=13)
    plt.tight_layout()
    out_path = out_dir / "scatter.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Scatter plot saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MultiResHuBERT")
    parser.add_argument("--config",      required=True, help="YAML config used for fine-tuning")
    parser.add_argument("--checkpoint",  required=True, help="Path to fine-tuning checkpoint .pt")
    parser.add_argument("--output-dir",  default="results/eval")
    parser.add_argument("--ablation",    action="store_true",
                        help="Run hi-res vs lo-res head ablation study")
    parser.add_argument("--plot",        action="store_true",
                        help="Save scatter-plot grid (requires matplotlib)")
    parser.add_argument("--device",      default=None,
                        help="Override device (e.g. cpu, cuda:1)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device or cfg["training"].get("device", "cpu")
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── build model & data ────────────────────────────────────────────────
    model  = load_model(cfg, args.checkpoint, device)
    loader = build_test_loader(cfg)

    # ── main inference ────────────────────────────────────────────────────
    log.info("Running inference …")
    result  = run_inference(model, loader, device)
    metrics = compute_metrics(result["preds"], result["labels"])

    # ── print main table ──────────────────────────────────────────────────
    print_main_table(metrics)
    print(
        f"\nN={len(result['utt_ids'])}  device={device}  ckpt={args.checkpoint}"
    )

    # ── ablation study ────────────────────────────────────────────────────
    if args.ablation:
        log.info("Running hi-res ablation (H₂ branch zeroed) …")
        preds_hi = ablated_forward(model, loader, device, ablate="lo")
        log.info("Running lo-res ablation (H₃ branch zeroed) …")
        preds_lo = ablated_forward(model, loader, device, ablate="hi")

        m_combined = metrics
        m_hi_only  = compute_metrics(preds_hi, result["labels"])
        m_lo_only  = compute_metrics(preds_lo, result["labels"])

        print_ablation_table(m_combined, m_hi_only, m_lo_only)

        save_metrics_csv(m_hi_only,  out_dir / "metrics_hi_only.csv")
        save_metrics_csv(m_lo_only,  out_dir / "metrics_lo_only.csv")

    # ── plots ─────────────────────────────────────────────────────────────
    if args.plot:
        plot_scatter(result["preds"], result["labels"], metrics, out_dir)

    # ── save outputs ──────────────────────────────────────────────────────
    save_predictions_csv(
        result["preds"], result["labels"], result["utt_ids"],
        out_dir / "predictions.csv",
    )
    save_metrics_csv(metrics, out_dir / "metrics.csv")

    # ── save pooled representations for downstream probing ────────────────
    np.save(out_dir / "h3_pooled.npy", result["h3_pooled"])
    np.save(out_dir / "h2_pooled.npy", result["h2_pooled"])
    log.info(
        "Saved pooled H₃ (%s) and H₂ (%s) to %s",
        result["h3_pooled"].shape, result["h2_pooled"].shape, out_dir,
    )


if __name__ == "__main__":
    main()
