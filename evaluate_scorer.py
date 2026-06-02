"""
Evaluate a trained PronunciationScorer on the Speechocean762 test split.

Usage:
    # Full evaluation
    python evaluate_scorer.py --config configs/scoring_config.yaml \
                               --checkpoint checkpoints/scorer/best_model.pt

    # Sanity-check shapes and checkpoint loading (no dataset needed)
    python evaluate_scorer.py --sanity-check --config configs/scoring_config.yaml \
                               --checkpoint checkpoints/scorer/best_model.pt

Results table (scores in [0, 10]):
    Dimension    | PCC    | MSE    | MAE
    -------------|--------|--------|--------
    total        |        |        |
    accuracy     |        |        |
    fluency      |        |        |
    prosodic     |        |        |
    MEAN         |        |        |

Ablation (text path zeroed):
    Dimension    | PCC(full) | PCC(speech-only) | Delta
    ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.speechocean_asr import SpeechOceanASRDataset, asr_collate_fn, SCORE_KEYS
from model.pronunciation_scorer import PronunciationScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCORE_MAX = 10.0


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
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:        PronunciationScorer,
    loader:       DataLoader,
    device:       torch.device,
    speech_only:  bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model over the entire loader.

    Returns:
        preds:   [N, 5] in [0, 1]  (model output before ×10)
        targets: [N, 5] in [0, 1]  (dataset labels)
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
    preds:   np.ndarray,   # [N, 5] in [0, 1]
    targets: np.ndarray,   # [N, 5] in [0, 1]
    scale:   float = SCORE_MAX,
) -> List[Dict[str, float]]:
    """Per-dimension PCC, MSE, MAE on the [0, 10] scale."""
    rows = []
    for i in range(preds.shape[1]):
        p = preds[:, i] * scale
        t = targets[:, i] * scale
        pcc = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-6 else float("nan")
        mse = float(np.mean((p - t) ** 2))
        mae = float(np.mean(np.abs(p - t)))
        rows.append({"pcc": pcc, "mse": mse, "mae": mae})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Table printing
# ─────────────────────────────────────────────────────────────────────────────

def _print_results(title: str, rows: List[Dict], dims: List[str] = SCORE_KEYS) -> None:
    W = 58
    print(f"\n=== {title} ===\n")
    print(f"{'Dimension':<14} | {'PCC':>7} | {'MSE':>7} | {'MAE':>7}")
    print("-" * W)
    for dim, r in zip(dims, rows):
        pcc = f"{r['pcc']:+.4f}" if not np.isnan(r["pcc"]) else "    nan"
        print(f"{dim:<14} | {pcc:>7} | {r['mse']:>7.4f} | {r['mae']:>7.4f}")
    mean_pcc = np.nanmean([r["pcc"] for r in rows])
    mean_mse = np.mean([r["mse"] for r in rows])
    mean_mae = np.mean([r["mae"] for r in rows])
    print("-" * W)
    print(f"{'MEAN':<14} | {mean_pcc:>+7.4f} | {mean_mse:>7.4f} | {mean_mae:>7.4f}")


def _print_ablation(
    full_rows:    List[Dict],
    speech_rows:  List[Dict],
    dims:         List[str] = SCORE_KEYS,
) -> None:
    W = 56
    print("\n=== ABLATION: text path contribution ===\n")
    print(f"{'Dimension':<14} | {'Full PCC':>9} | {'Speech-only':>11} | {'Delta':>7}")
    print("-" * W)
    for dim, fr, sr in zip(dims, full_rows, speech_rows):
        fp  = fr["pcc"]
        sp  = sr["pcc"]
        dlt = fp - sp if not (np.isnan(fp) or np.isnan(sp)) else float("nan")
        fp_s  = f"{fp:+.4f}" if not np.isnan(fp)  else "     nan"
        sp_s  = f"{sp:+.4f}" if not np.isnan(sp)  else "     nan"
        dl_s  = f"{dlt:+.4f}" if not np.isnan(dlt) else "     nan"
        print(f"{dim:<14} | {fp_s:>9} | {sp_s:>11} | {dl_s:>7}")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(cfg: dict, checkpoint: Optional[str]) -> None:
    """
    Confirm:
    1. Model builds without error.
    2. Forward pass produces correct shapes.
    3. Checkpoint loads cleanly.
    4. Scores are in [0, 1].

    Runs on CPU with random tensors — no dataset required.
    """
    print("\n=== SANITY CHECK ===\n")
    device = torch.device("cpu")

    log.info("Building model on CPU …")
    model = PronunciationScorer(cfg).to(device).eval()

    if checkpoint:
        log.info("Loading checkpoint: %s …", checkpoint)
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        log.info("  Checkpoint loaded.")
        print(f"  [OK]  checkpoint loaded from {checkpoint}")

    # Random inputs.
    B, T = 2, 16000   # 1-second batch of 2
    waveforms   = torch.randn(B, T)
    attn_mask   = torch.ones(B, T, dtype=torch.long)
    transcripts = ["THE SPEAKER SAID HELLO", "THIS IS A TEST"]

    with torch.no_grad():
        scores = model(waveforms, attn_mask, transcripts)

    assert scores.shape == (B, 5), f"Expected ({B}, 5), got {tuple(scores.shape)}"
    assert scores.min().item() >= 0.0, f"Score below 0: {scores.min()}"
    assert scores.max().item() <= 1.0, f"Score above 1: {scores.max()}"

    print(f"  [OK]  forward pass: input [B={B}, T={T}] → scores {tuple(scores.shape)}")
    print(f"  [OK]  score range: [{scores.min().item():.4f}, {scores.max().item():.4f}] (in [0,1])")
    print(f"  [OK]  SCORE_KEYS: {SCORE_KEYS}")
    print("\nSanity check passed — model is ready for training / evaluation.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PronunciationScorer")
    parser.add_argument("--config",       required=True)
    parser.add_argument("--checkpoint",   default=None)
    parser.add_argument("--split",        default=None)
    parser.add_argument("--batch-size",   type=int, default=None)
    parser.add_argument("--device",       default=None)
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run shape/checkpoint sanity check and exit.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    if args.sanity_check:
        sanity_check(cfg, args.checkpoint)
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required for full evaluation.")

    dcfg       = cfg["data"]
    split      = args.split or dcfg.get("test_split", "test")
    device     = torch.device(args.device or cfg["training"].get("device", "cpu"))
    batch_size = args.batch_size or dcfg.get("batch_size", 16)

    log.info("Loading %s split …", split)
    ds = SpeechOceanASRDataset(cfg, split=split)
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = dcfg.get("num_workers", 4),
        collate_fn  = asr_collate_fn,
        pin_memory  = device.type == "cuda",
    )
    log.info("  %d utterances", len(ds))

    log.info("Loading checkpoint: %s …", args.checkpoint)
    model = PronunciationScorer(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    # Full inference
    log.info("Running full Fusion-C inference …")
    full_preds, targets = run_inference(model, loader, device, speech_only=False)
    full_metrics = compute_metrics(full_preds, targets)
    _print_results("RESULTS", full_metrics)

    # Speech-only ablation
    log.info("Running speech-only ablation (text_emb = 0) …")
    speech_preds, _ = run_inference(model, loader, device, speech_only=True)
    speech_metrics  = compute_metrics(speech_preds, targets)
    _print_ablation(full_metrics, speech_metrics)

    print()


if __name__ == "__main__":
    main()
