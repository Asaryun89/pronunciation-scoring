from __future__ import annotations

"""
Training entry point for HubertScoringModel on SpeechOcean762.

CLI usage examples::

    python -m models.train --output-dir runs/exp1
    python -m models.train --epochs 15 --batch-size 8 --lr 2e-5
"""

# Allow `python models/train.py` to be run directly from the repo root
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from models.scoring_model import HubertScoringModel
from utils.dataset import SpeechOcean762Dataset, collate_fn

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """All hyper-parameters for a training run."""

    # Model
    model_name: str = "facebook/hubert-base-ls960"
    d_model: int = 256
    num_heads: int = 8
    num_audio_transformer_layers: int = 1
    num_fusion_transformer_layers: int = 2
    mlp_hidden_layers: int = 2
    dropout: float = 0.1
    num_unfreeze_hubert_layers: int = 12

    # Optimisation
    epochs: int = 10
    batch_size: int = 4
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0

    # Loss weights
    lambda_sent: float = 1.0
    lambda_prosody: float = 0.5

    # Data
    max_audio_seconds: float = 20.0
    num_workers: int = 2

    # Runtime
    output_dir: str = "runs/exp1"
    seed: int = 42
    log_every: int = 50
    fp16: bool = True


# ─── Seeding ──────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Seed Python random, numpy, torch, and torch.cuda for reproducibility.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Loss ─────────────────────────────────────────────────────────────────────


class ScoringLoss(nn.Module):
    """Combined Huber (Smooth L1) loss for sentence scores and prosody features.

    Prosody targets are z-score standardised per batch per feature before the
    loss is computed, so the prosody head can learn relative patterns without
    being dominated by scale differences between features.

    Args:
        lambda_sent: Weight for the sentence-level scoring loss.
        lambda_prosody: Weight for the prosody auxiliary loss.
    """

    def __init__(self, lambda_sent: float = 1.0, lambda_prosody: float = 0.5) -> None:
        super().__init__()
        self.lambda_sent = lambda_sent
        self.lambda_prosody = lambda_prosody
        self.huber = nn.SmoothL1Loss()

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        sent_targets: torch.Tensor,
        prosody_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the combined loss.

        Args:
            preds: Model output dict containing ``"sent_pred"`` and
                ``"prosody_pred"``.
            sent_targets: ``(B, 5)`` normalised sentence scores in ``[0, 1]``.
            prosody_targets: ``(B, 5)`` raw prosody feature tensor.

        Returns:
            Tuple of (total loss tensor, dict with scalar components
            ``total``, ``sent``, ``prosody``).
        """
        sent_loss = self.huber(preds["sent_pred"], sent_targets)

        mean = prosody_targets.mean(dim=0, keepdim=True)
        std = prosody_targets.std(dim=0, keepdim=True) + 1e-8
        prosody_norm = (prosody_targets - mean) / std
        prosody_loss = self.huber(preds["prosody_pred"], prosody_norm)

        total = self.lambda_sent * sent_loss + self.lambda_prosody * prosody_loss

        return total, {
            "total": total.item(),
            "sent": sent_loss.item(),
            "prosody": prosody_loss.item(),
        }


# ─── Metrics ──────────────────────────────────────────────────────────────────


def compute_correlations(
    preds: np.ndarray, targets: np.ndarray
) -> Dict[str, List[float]]:
    """Compute per-aspect Pearson correlation, Spearman correlation, and MAE.

    Columns with near-zero variance in either array return correlation of 0.0.

    Args:
        preds: ``(N, num_aspects)`` array of predictions.
        targets: ``(N, num_aspects)`` array of targets.

    Returns:
        Dict with keys ``"pearson"``, ``"spearman"``, ``"mae"`` — each a list
        of ``num_aspects`` floats.
    """
    num_aspects = preds.shape[1]
    pearson_vals: List[float] = []
    spearman_vals: List[float] = []
    mae_vals: List[float] = []

    for i in range(num_aspects):
        p, t = preds[:, i], targets[:, i]
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            pearson_vals.append(0.0)
            spearman_vals.append(0.0)
        else:
            pearson_vals.append(float(pearsonr(p, t)[0]))
            spearman_vals.append(float(spearmanr(p, t)[0]))
        mae_vals.append(float(np.mean(np.abs(p - t))))

    return {"pearson": pearson_vals, "spearman": spearman_vals, "mae": mae_vals}


# ─── Training loop ────────────────────────────────────────────────────────────


def train_one_epoch(
    model: HubertScoringModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: ScoringLoss,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    log_every: int,
    grad_clip: float,
) -> Dict[str, float]:
    """Run one full training epoch.

    Performs gradient clipping, AMP scaling, and scheduler stepping once per
    batch.  Logs running averages every ``log_every`` steps.

    Args:
        model: The scoring model in training mode.
        loader: Training DataLoader.
        optimizer: Optimizer (AdamW).
        scheduler: LR scheduler stepped after every batch.
        criterion: Combined scoring loss.
        scaler: AMP GradScaler (may be disabled on CPU).
        device: Compute device.
        log_every: Log interval in steps.
        grad_clip: Maximum gradient norm.

    Returns:
        Dict with average ``total``, ``sent``, and ``prosody`` loss values for
        the epoch.
    """
    model.train()
    totals: Dict[str, float] = {"total": 0.0, "sent": 0.0, "prosody": 0.0}

    for step, batch in enumerate(loader, 1):
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        phoneme_ids = batch["phoneme_ids"].to(device)
        phoneme_mask = batch["phoneme_mask"].to(device)
        sent_scores = batch["sent_scores"].to(device)
        prosody_feats = batch["prosody_feats"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            preds = model(input_values, phoneme_ids, attention_mask, phoneme_mask)
            loss, components = criterion(preds, sent_scores, prosody_feats)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for k, v in components.items():
            totals[k] = totals.get(k, 0.0) + v

        if step % log_every == 0:
            avg = {k: v / step for k, v in totals.items()}
            logger.info(
                "  step %d  loss=%.4f  sent=%.4f  prosody=%.4f",
                step, avg["total"], avg["sent"], avg["prosody"],
            )

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: HubertScoringModel,
    loader: DataLoader,
    criterion: ScoringLoss,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Evaluate the model on a DataLoader.

    Args:
        model: The scoring model in eval mode.
        loader: Evaluation DataLoader.
        criterion: Combined scoring loss.
        device: Compute device.

    Returns:
        Tuple of:

        - ``loss_dict``: Average ``total``, ``sent``, ``prosody`` losses.
        - ``metrics_dict``: Per-aspect ``pearson``, ``spearman``, ``mae`` lists
          plus scalar ``pcc_total``, ``scc_total``, ``mae_total``.
    """
    model.eval()
    totals: Dict[str, float] = {"total": 0.0, "sent": 0.0, "prosody": 0.0}
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        phoneme_ids = batch["phoneme_ids"].to(device)
        phoneme_mask = batch["phoneme_mask"].to(device)
        sent_scores = batch["sent_scores"].to(device)
        prosody_feats = batch["prosody_feats"].to(device)

        preds = model(input_values, phoneme_ids, attention_mask, phoneme_mask)
        _, components = criterion(preds, sent_scores, prosody_feats)

        for k, v in components.items():
            totals[k] = totals.get(k, 0.0) + v

        all_preds.append(preds["sent_pred"].cpu().float().numpy())
        all_targets.append(sent_scores.cpu().float().numpy())

    n = len(loader)
    loss_dict = {k: v / n for k, v in totals.items()}

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    corr = compute_correlations(preds_arr, targets_arr)

    # SCORE_KEYS order: accuracy, completeness, fluency, prosody, total → index 4
    total_idx = 4
    metrics: Dict[str, Any] = {
        **corr,
        "pcc_total": corr["pearson"][total_idx],
        "scc_total": corr["spearman"][total_idx],
        "mae_total": corr["mae"][total_idx],
    }
    return loss_dict, metrics


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> TrainConfig:
    """Parse command-line arguments and return a populated :class:`TrainConfig`.

    Returns:
        TrainConfig with CLI overrides applied.
    """
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train HubertScoringModel on SpeechOcean762",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=cfg.output_dir,
                        help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=cfg.epochs,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=cfg.learning_rate,
                        help="Peak learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay,
                        help="AdamW weight decay")
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers,
                        help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=cfg.seed,
                        help="Random seed")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable automatic mixed precision (FP16)")
    parser.add_argument("--model-name", default=cfg.model_name,
                        help="HuggingFace model ID for HuBERT backbone")

    args = parser.parse_args()
    cfg.output_dir = args.output_dir
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed
    cfg.fp16 = not args.no_fp16
    cfg.model_name = args.model_name
    return cfg


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Full training pipeline: data → model → train/eval loop → checkpointing."""
    cfg = parse_args()
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg.fp16 and device.type == "cuda"

    logger.info("Loading datasets…")
    train_ds = SpeechOcean762Dataset("train", cfg.max_audio_seconds)
    test_ds = SpeechOcean762Dataset("test", cfg.max_audio_seconds)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
    )

    logger.info("Building model…")
    model = HubertScoringModel(
        model_name=cfg.model_name,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_audio_transformer_layers=cfg.num_audio_transformer_layers,
        num_fusion_transformer_layers=cfg.num_fusion_transformer_layers,
        mlp_hidden_layers=cfg.mlp_hidden_layers,
        dropout=cfg.dropout,
        num_unfreeze_hubert_layers=cfg.num_unfreeze_hubert_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total params: %d | Trainable: %d", total_params, trainable_params)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    criterion = ScoringLoss(cfg.lambda_sent, cfg.lambda_prosody)

    csv_path = output_dir / "metrics.csv"
    csv_columns = [
        "epoch", "train_loss", "train_sent_loss", "train_prosody_loss",
        "test_loss", "test_sent_loss", "test_prosody_loss",
        "pcc_total", "scc_total", "mae_total",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()

    best_pcc = -1.0

    try:
        for epoch in range(1, cfg.epochs + 1):
            logger.info("=== Epoch %d/%d ===", epoch, cfg.epochs)

            train_losses = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion,
                scaler, device, cfg.log_every, cfg.grad_clip,
            )
            test_losses, metrics = evaluate(model, test_loader, criterion, device)

            pcc_total = float(metrics["pcc_total"])
            row = {
                "epoch": epoch,
                "train_loss": f"{train_losses['total']:.6f}",
                "train_sent_loss": f"{train_losses['sent']:.6f}",
                "train_prosody_loss": f"{train_losses['prosody']:.6f}",
                "test_loss": f"{test_losses['total']:.6f}",
                "test_sent_loss": f"{test_losses['sent']:.6f}",
                "test_prosody_loss": f"{test_losses['prosody']:.6f}",
                "pcc_total": f"{pcc_total:.4f}",
                "scc_total": f"{float(metrics['scc_total']):.4f}",
                "mae_total": f"{float(metrics['mae_total']):.4f}",
            }
            writer.writerow(row)
            csv_file.flush()

            logger.info(
                "Epoch %d | train=%.4f | test=%.4f | pcc_total=%.4f",
                epoch, train_losses["total"], test_losses["total"], pcc_total,
            )

            if pcc_total > best_pcc:
                best_pcc = pcc_total
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "test_metrics": {k: v for k, v in metrics.items()},
                    "pcc_total": pcc_total,
                }
                torch.save(ckpt, output_dir / "best_checkpoint.pt")
                logger.info("  New best checkpoint (pcc_total=%.4f)", best_pcc)
    finally:
        csv_file.close()

    logger.info("Training complete. Best pcc_total=%.4f", best_pcc)


if __name__ == "__main__":
    main()
