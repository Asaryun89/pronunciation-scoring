from __future__ import annotations

"""Training entry point for HubertScoringModel (BGE text branch) on SpeechOcean762.

Usage::

    python -m models.train --output-dir runs/bge_exp1
    python -m models.train --epochs 15 --batch-size 8 --no-freeze-bge
"""

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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from models.scoring_model import HubertScoringModel
from utils.dataset import SpeechOcean762Dataset, collate_fn

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Model
    hubert_model_name: str = "facebook/hubert-base-ls960"
    bge_model_name: str = "BAAI/bge-small-en-v1.5"
    d_model: int = 256
    num_heads: int = 8
    num_audio_transformer_layers: int = 1
    num_fusion_transformer_layers: int = 2
    mlp_hidden_layers: int = 2
    dropout: float = 0.1
    num_unfreeze_hubert_layers: int = 12
    freeze_bge: bool = True
    # Optimisation
    epochs: int = 10
    batch_size: int = 4
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    # Loss
    lambda_sent: float = 1.0
    lambda_prosody: float = 0.5
    # Data
    max_audio_seconds: float = 20.0
    max_text_length: int = 128
    num_workers: int = 2
    # Runtime
    output_dir: str = "runs/bge_exp1"
    seed: int = 42
    log_every: int = 50
    fp16: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ScoringLoss(nn.Module):
    """Smooth L1 for sentence scores + z-normalised prosody features."""

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
        sent_loss = self.huber(preds["sent_pred"], sent_targets)
        mean = prosody_targets.mean(dim=0, keepdim=True)
        std = prosody_targets.std(dim=0, keepdim=True) + 1e-8
        prosody_loss = self.huber(preds["prosody_pred"], (prosody_targets - mean) / std)
        total = self.lambda_sent * sent_loss + self.lambda_prosody * prosody_loss
        return total, {"total": total.item(), "sent": sent_loss.item(), "prosody": prosody_loss.item()}


def compute_correlations(preds: np.ndarray, targets: np.ndarray) -> Dict[str, List[float]]:
    """Per-aspect Pearson, Spearman, MAE."""
    pearson_vals, spearman_vals, mae_vals = [], [], []
    for i in range(preds.shape[1]):
        p, t = preds[:, i], targets[:, i]
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            pearson_vals.append(0.0)
            spearman_vals.append(0.0)
        else:
            pearson_vals.append(float(pearsonr(p, t)[0]))
            spearman_vals.append(float(spearmanr(p, t)[0]))
        mae_vals.append(float(np.mean(np.abs(p - t))))
    return {"pearson": pearson_vals, "spearman": spearman_vals, "mae": mae_vals}


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
    model.train()
    totals: Dict[str, float] = {"total": 0.0, "sent": 0.0, "prosody": 0.0}

    for step, batch in enumerate(loader, 1):
        input_values = batch["input_values"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        audio_attention_mask = batch["audio_attention_mask"].to(device)
        sent_scores = batch["sent_scores"].to(device)
        prosody_feats = batch["prosody_feats"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            preds = model(
                input_values=input_values,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                audio_attention_mask=audio_attention_mask,
            )
            loss, components = criterion(preds, sent_scores, prosody_feats)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for k, v in components.items():
            totals[k] += v

        if step % log_every == 0:
            avg = {k: v / step for k, v in totals.items()}
            logger.info("  step %d  loss=%.4f  sent=%.4f  prosody=%.4f",
                        step, avg["total"], avg["sent"], avg["prosody"])

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: HubertScoringModel,
    loader: DataLoader,
    criterion: ScoringLoss,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    model.eval()
    totals: Dict[str, float] = {"total": 0.0, "sent": 0.0, "prosody": 0.0}
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for batch in loader:
        input_values = batch["input_values"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        audio_attention_mask = batch["audio_attention_mask"].to(device)
        sent_scores = batch["sent_scores"].to(device)
        prosody_feats = batch["prosody_feats"].to(device)

        preds = model(
            input_values=input_values,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            audio_attention_mask=audio_attention_mask,
        )
        _, components = criterion(preds, sent_scores, prosody_feats)
        for k, v in components.items():
            totals[k] += v
        all_preds.append(preds["sent_pred"].cpu().float().numpy())
        all_targets.append(sent_scores.cpu().float().numpy())

    loss_dict = {k: v / len(loader) for k, v in totals.items()}
    corr = compute_correlations(np.concatenate(all_preds), np.concatenate(all_targets))
    return loss_dict, {**corr, "pcc_total": corr["pearson"][4], "scc_total": corr["spearman"][4], "mae_total": corr["mae"][4]}


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(
        description="Train HubertScoringModel (BGE) on SpeechOcean762",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default=cfg.output_dir)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--hubert-model-name", default=cfg.hubert_model_name)
    p.add_argument("--bge-model-name", default=cfg.bge_model_name)
    p.add_argument("--no-freeze-bge", action="store_true")

    args = p.parse_args()
    cfg.output_dir = args.output_dir
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed
    cfg.fp16 = not args.no_fp16
    cfg.hubert_model_name = args.hubert_model_name
    cfg.bge_model_name = args.bge_model_name
    cfg.freeze_bge = not args.no_freeze_bge
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg.fp16 and device.type == "cuda"

    logger.info("Loading datasets…")
    train_ds = SpeechOcean762Dataset("train", cfg.max_audio_seconds, cfg.bge_model_name, cfg.max_text_length)
    test_ds  = SpeechOcean762Dataset("test",  cfg.max_audio_seconds, cfg.bge_model_name, cfg.max_text_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,      shuffle=True,  num_workers=cfg.num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

    logger.info("Building model…")
    model = HubertScoringModel(
        hubert_model_name=cfg.hubert_model_name,
        bge_model_name=cfg.bge_model_name,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_audio_transformer_layers=cfg.num_audio_transformer_layers,
        num_fusion_transformer_layers=cfg.num_fusion_transformer_layers,
        mlp_hidden_layers=cfg.mlp_hidden_layers,
        dropout=cfg.dropout,
        num_unfreeze_hubert_layers=cfg.num_unfreeze_hubert_layers,
        freeze_bge=cfg.freeze_bge,
    ).to(device)

    def _count(params): return sum(p.numel() for p in params)
    hubert_p  = list(model.hubert.parameters())
    bge_p     = list(model.text_embedder.encoder.parameters())
    other_p   = [p for n, p in model.named_parameters() if not n.startswith(("hubert.", "text_embedder.encoder."))]
    logger.info("HuBERT %5.1fM (trainable %.1fM) | BGE %5.1fM (trainable %.1fM) | Other %.1fM | Total trainable %.1fM",
                _count(hubert_p)/1e6, _count(p for p in hubert_p if p.requires_grad)/1e6,
                _count(bge_p)/1e6,    _count(p for p in bge_p    if p.requires_grad)/1e6,
                _count(other_p)/1e6,  _count(p for p in model.parameters() if p.requires_grad)/1e6)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    total_steps  = len(train_loader) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_fp16)
    criterion = ScoringLoss(cfg.lambda_sent, cfg.lambda_prosody)

    csv_columns = ["epoch", "train_loss", "train_sent_loss", "train_prosody_loss",
                   "test_loss", "test_sent_loss", "test_prosody_loss", "pcc_total", "scc_total", "mae_total"]
    csv_file = open(output_dir / "metrics.csv", "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()

    best_pcc = -1.0
    try:
        for epoch in range(1, cfg.epochs + 1):
            logger.info("=== Epoch %d/%d ===", epoch, cfg.epochs)
            train_losses = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, cfg.log_every, cfg.grad_clip)
            test_losses, metrics = evaluate(model, test_loader, criterion, device)
            pcc_total = float(metrics["pcc_total"])

            writer.writerow({
                "epoch": epoch,
                "train_loss": f"{train_losses['total']:.6f}", "train_sent_loss": f"{train_losses['sent']:.6f}", "train_prosody_loss": f"{train_losses['prosody']:.6f}",
                "test_loss":  f"{test_losses['total']:.6f}",  "test_sent_loss":  f"{test_losses['sent']:.6f}",  "test_prosody_loss":  f"{test_losses['prosody']:.6f}",
                "pcc_total": f"{pcc_total:.4f}", "scc_total": f"{float(metrics['scc_total']):.4f}", "mae_total": f"{float(metrics['mae_total']):.4f}",
            })
            csv_file.flush()
            logger.info("Epoch %d | train=%.4f | test=%.4f | pcc_total=%.4f", epoch, train_losses["total"], test_losses["total"], pcc_total)

            if pcc_total > best_pcc:
                best_pcc = pcc_total
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": asdict(cfg), "test_metrics": dict(metrics), "pcc_total": pcc_total},
                           output_dir / "best_checkpoint.pt")
                logger.info("  New best checkpoint (pcc_total=%.4f)", best_pcc)
    finally:
        csv_file.close()

    logger.info("Training complete. Best pcc_total=%.4f", best_pcc)


if __name__ == "__main__":
    main()
