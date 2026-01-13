from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

from data.collate import collate_fn
from data.dataset import DummyPronunciationDataset
from features.mfcc import MFCCConfig
from models.pronunciation_model import ModelConfig, PronunciationModel
from training.loss import PronunciationLoss
from utils.audio import AudioConfig
from utils.logger import get_logger
from utils.metrics import mae, rmse


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict[str, Any]) -> PronunciationModel:
    n_mfcc = cfg["feature"]["n_mfcc"]
    input_dim = n_mfcc * 3 if cfg["feature"]["add_deltas"] else n_mfcc
    model_cfg = ModelConfig(
        input_dim=input_dim,
        cnn_channels=cfg["model"]["cnn_channels"],
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
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
    feat_cfg = MFCCConfig(
        sample_rate=cfg["sample_rate"],
        n_mfcc=cfg["feature"]["n_mfcc"],
        n_fft=cfg["feature"]["n_fft"],
        hop_length=cfg["feature"]["hop_length"],
        win_length=cfg["feature"]["win_length"],
        add_deltas=cfg["feature"]["add_deltas"],
    )

    train_set = DummyPronunciationDataset(
        size=cfg["data"]["train_size"],
        audio_cfg=audio_cfg,
        feat_cfg=feat_cfg,
        num_phonemes=cfg["data"]["num_phonemes"],
        max_phoneme_frames=cfg["data"]["max_phoneme_frames"],
    )
    val_set = DummyPronunciationDataset(
        size=cfg["data"]["val_size"],
        audio_cfg=audio_cfg,
        feat_cfg=feat_cfg,
        num_phonemes=cfg["data"]["num_phonemes"],
        max_phoneme_frames=cfg["data"]["max_phoneme_frames"],
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
    criterion = PronunciationLoss()
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
            segments = batch["phoneme_segments"].to(device)
            seg_lengths = batch["seg_lengths"].to(device)
            scores = batch["scores"].to(device)
            phoneme_scores = batch["phoneme_scores"].to(device)

            out = model(features, feat_lengths, segments, seg_lengths)
            loss = criterion(
                out["utterance_score"],
                scores,
                out["phoneme_scores"],
                phoneme_scores,
                seg_lengths,
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
                segments = batch["phoneme_segments"].to(device)
                seg_lengths = batch["seg_lengths"].to(device)
                scores = batch["scores"].to(device)
                out = model(features, feat_lengths, segments, seg_lengths)
                all_pred.append(out["utterance_score"].cpu())
                all_target.append(scores.cpu())
            pred = torch.cat(all_pred)
            target = torch.cat(all_target)
            logger.info("val_mae=%.3f val_rmse=%.3f", mae(pred, target), rmse(pred, target))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg}, output_path)
    logger.info("saved checkpoint to %s", output_path)


if __name__ == "__main__":
    main()
