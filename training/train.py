from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

from data.collate import collate_fn
from data.dataset import DummyPronunciationDataset
from features.logmel import LogMelConfig
from models.pronunciation_model import ModelConfig, PronunciationModel
from training.loss import PronunciationLoss
from utils.audio import AudioConfig
from utils.logger import get_logger
from utils.metrics import mae, rmse


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict[str, Any]) -> PronunciationModel:
    n_mels = cfg["feature"]["n_mels"]
    input_dim = n_mels * 3 if cfg["feature"]["add_deltas"] else n_mels
    model_cfg = ModelConfig(
        input_dim=input_dim,
        ssl_dim=cfg["model"]["ssl_dim"],
        pron_dim=cfg["model"]["pron_dim"],
        transformer_heads=cfg["model"]["transformer_heads"],
        transformer_layers=cfg["model"]["transformer_layers"],
        dropout=cfg["model"]["dropout"],
    )
    return PronunciationModel(model_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--output", type=str, default="checkpoints/pronunciation.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_cfg = AudioConfig(
        sample_rate=cfg["sample_rate"],
        max_audio_seconds=cfg["max_audio_seconds"],
    )
    feat_cfg = LogMelConfig(
        sample_rate=cfg["sample_rate"],
        n_mels=cfg["feature"]["n_mels"],
        n_fft=cfg["feature"]["n_fft"],
        hop_length=cfg["feature"]["hop_length"],
        win_length=cfg["feature"]["win_length"],
        add_deltas=cfg["feature"]["add_deltas"],
    )

    train_set = DummyPronunciationDataset(
        size=cfg["data"]["train_size"],
        audio_cfg=audio_cfg,
        feat_cfg=feat_cfg,
        phoneme_vocab_size=cfg["data"]["phoneme_vocab_size"],
        min_phonemes=cfg["data"]["min_phonemes"],
        max_phonemes=cfg["data"]["max_phonemes"],
    )
    val_set = DummyPronunciationDataset(
        size=cfg["data"]["val_size"],
        audio_cfg=audio_cfg,
        feat_cfg=feat_cfg,
        phoneme_vocab_size=cfg["data"]["phoneme_vocab_size"],
        min_phonemes=cfg["data"]["min_phonemes"],
        max_phonemes=cfg["data"]["max_phonemes"],
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_model(cfg).to(device)
    criterion = PronunciationLoss(
        utterance_weight=cfg["training"]["loss_weights"]["utterance"],
        phoneme_weight=cfg["training"]["loss_weights"]["phoneme"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    logger = get_logger("train")

    for epoch in range(cfg["training"]["num_epochs"]):
        model.train()
        for step, batch in enumerate(train_loader):
            features = batch["features"].to(device)
            feat_lengths = batch["feat_lengths"].to(device)
            phoneme_lengths = batch["phoneme_lengths"].to(device)
            phoneme_scores = batch["phoneme_scores"].to(device)
            utterance_scores = batch["utterance_scores"].to(device)

            out = model(features, feat_lengths, phoneme_lengths)
            loss = criterion(
                out["utterance_score"],
                utterance_scores,
                out["phoneme_scores"],
                phoneme_scores,
                out["phoneme_mask"],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % cfg["training"]["log_every"] == 0:
                logger.info("epoch=%d step=%d loss=%.4f", epoch, step, loss.item())

        model.eval()
        with torch.no_grad():
            all_pred = []
            all_target = []
            for batch in val_loader:
                features = batch["features"].to(device)
                feat_lengths = batch["feat_lengths"].to(device)
                phoneme_lengths = batch["phoneme_lengths"].to(device)
                utterance_scores = batch["utterance_scores"].to(device)
                out = model(features, feat_lengths, phoneme_lengths)
                all_pred.append(out["utterance_score"].cpu())
                all_target.append(utterance_scores.cpu())
            pred = torch.cat(all_pred)
            target = torch.cat(all_target)
            logger.info("val_mae=%.3f val_rmse=%.3f", mae(pred, target), rmse(pred, target))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg}, output_path)
    logger.info("saved checkpoint to %s", output_path)


if __name__ == "__main__":
    main()
