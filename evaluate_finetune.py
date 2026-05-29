#!/usr/bin/env python3
"""
Evaluate a trained Fusion-C checkpoint on the Speechocean762 test split.

Prints per-dimension PCC / MSE / MAE, then an ablation table comparing
speech-only (text embedding zeroed) vs. full Fusion-C.

Usage
─────
    cd multi_res_hubert
    python evaluate_finetune.py \\
        --config    configs/finetune_config.yaml \\
        --checkpoint checkpoints/finetune/best_model.pt

    # Override split (default: test_split from config)
    python evaluate_finetune.py --config ... --checkpoint ... --split test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.speechocean_dataset import (
    FUSION_SCORE_KEYS,
    SpeechoceanFusionDataset,
    fusion_collate_fn,
)
from model.multireshubert_finetune import MultiResHuBERTFinetune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCORE_MAX = 10.0   # for display — predictions are in [0, 1], multiply for [0, 10]


# ─────────────────────────────────────────────────────────────────────────────
# Config loader (mirrors finetune_fusionC.load_cfg)
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
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
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:       MultiResHuBERTFinetune,
    loader:      DataLoader,
    device:      torch.device,
    speech_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model over entire loader.

    Returns:
        preds:   [N, 5] in [0, 1]
        targets: [N, 5] in [0, 1]
    """
    model.eval()
    all_pred:   List[np.ndarray] = []
    all_target: List[np.ndarray] = []

    for batch in loader:
        waveforms   = batch["waveforms"].to(device)
        attn_mask   = batch["attention_mask"].to(device)
        targets     = batch["labels"]
        transcripts = batch["transcripts"]

        pred = model(waveforms, attn_mask, transcripts, speech_only=speech_only)
        all_pred.append(pred.cpu().float().numpy())
        all_target.append(targets.float().numpy())

    return (
        np.concatenate(all_pred,   axis=0),
        np.concatenate(all_target, axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    preds:   np.ndarray,   # [N, 5] normalised [0, 1]
    targets: np.ndarray,   # [N, 5] normalised [0, 1]
    scale:   float = SCORE_MAX,
) -> List[Dict[str, float]]:
    """Per-dimension PCC, MSE, MAE.  Values are on the original [0, 10] scale."""
    rows = []
    for i in range(preds.shape[1]):
        p = preds[:, i] * scale
        t = targets[:, i] * scale
        pcc = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-6 else float("nan")
        rows.append({
            "pcc": pcc,
            "mse": float(np.mean((p - t) ** 2)),
            "mae": float(np.mean(np.abs(p - t))),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Table printing
# ─────────────────────────────────────────────────────────────────────────────

def _print_results_table(
    title:   str,
    rows:    List[Dict[str, float]],
    dims:    List[str] = FUSION_SCORE_KEYS,
) -> None:
    print(f"\n=== {title} ===\n")
    print(f"{'Dimension':<14}| {'PCC':>8} | {'MSE':>8} | {'MAE':>8}")
    print("-" * 44)
    for dim, r in zip(dims, rows):
        pcc = f"{r['pcc']:.4f}" if not np.isnan(r['pcc']) else "  nan"
        print(f"{dim:<14}| {pcc:>8} | {r['mse']:>8.4f} | {r['mae']:>8.4f}")
    mean_pcc = np.nanmean([r["pcc"] for r in rows])
    mean_mse = np.mean([r["mse"] for r in rows])
    mean_mae = np.mean([r["mae"] for r in rows])
    print("-" * 44)
    print(f"{'MEAN':<14}| {mean_pcc:>8.4f} | {mean_mse:>8.4f} | {mean_mae:>8.4f}")


def _print_ablation_table(
    speech_rows: List[Dict[str, float]],
    fusion_rows: List[Dict[str, float]],
    dims:        List[str] = FUSION_SCORE_KEYS,
) -> None:
    print("\n=== ABLATION: speech-only vs FusionC ===\n")
    print(
        f"{'Dimension':<14}| {'Speech-only PCC':>17} | "
        f"{'FusionC PCC':>13} | {'Delta':>7}"
    )
    print("-" * 57)
    for dim, sr, fr in zip(dims, speech_rows, fusion_rows):
        s_pcc = sr["pcc"]
        f_pcc = fr["pcc"]
        delta = f_pcc - s_pcc if not (np.isnan(s_pcc) or np.isnan(f_pcc)) else float("nan")
        s_str = f"{s_pcc:.4f}" if not np.isnan(s_pcc) else "   nan"
        f_str = f"{f_pcc:.4f}" if not np.isnan(f_pcc) else "   nan"
        d_str = f"{delta:+.4f}" if not np.isnan(delta) else "   nan"
        print(f"{dim:<14}| {s_str:>17} | {f_str:>13} | {d_str:>7}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Fusion-C model")
    parser.add_argument("--config",     required=True, help="YAML config path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint .pt")
    parser.add_argument("--split",      default=None,
                        help="Dataset split to evaluate on (default: test_split from config)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    dcfg  = cfg["data"]
    split = args.split or dcfg.get("test_split", "test")
    device = torch.device(
        args.device or cfg["training"].get("device", "cpu")
    )
    batch_size = args.batch_size or dcfg.get("batch_size", 16)

    # ── Dataset ───────────────────────────────────────────────────────────
    log.info("Loading %s split …", split)
    ds = SpeechoceanFusionDataset(cfg, split=split, augment=False)
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = dcfg.get("num_workers", 4),
        collate_fn  = fusion_collate_fn,
        pin_memory  = device.type == "cuda",
    )
    log.info("  %d utterances", len(ds))

    # ── Model ─────────────────────────────────────────────────────────────
    log.info("Loading checkpoint: %s …", args.checkpoint)
    model = MultiResHuBERTFinetune(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    log.info("Model loaded.")

    # ── Fusion-C inference ────────────────────────────────────────────────
    log.info("Running Fusion-C inference …")
    fusion_preds, targets = run_inference(model, loader, device, speech_only=False)
    fusion_metrics = compute_metrics(fusion_preds, targets)
    _print_results_table(
        "FUSION C RESULTS (bge-small-en-v1.5)",
        fusion_metrics,
    )

    # ── Speech-only ablation ──────────────────────────────────────────────
    log.info("Running speech-only ablation (text_emb = 0) …")
    speech_preds, _ = run_inference(model, loader, device, speech_only=True)
    speech_metrics  = compute_metrics(speech_preds, targets)
    _print_ablation_table(speech_metrics, fusion_metrics)

    print()


if __name__ == "__main__":
    main()
