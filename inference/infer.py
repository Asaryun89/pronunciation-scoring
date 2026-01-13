from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from features.mfcc import MFCCConfig, compute_mfcc, stack_frames
from models.pronunciation_model import ModelConfig, PronunciationModel
from utils.audio import AudioConfig, preprocess_audio


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


def uniform_segments(num_frames: int, num_segments: int) -> torch.Tensor:
    """Create simple uniform segments as a forced-alignment placeholder."""
    if num_segments <= 0:
        return torch.zeros((0, 2), dtype=torch.long)
    boundaries = np.linspace(0, num_frames, num_segments + 1).astype(int)
    segments = np.stack([boundaries[:-1], boundaries[1:]], axis=1)
    segments[:, 1] = np.maximum(segments[:, 1], segments[:, 0] + 1)
    return torch.tensor(segments, dtype=torch.long)


def infer(
    audio_path: str,
    ref_text: str,
    config_path: str,
    checkpoint_path: Optional[str] = None,
    phoneme_segments: Optional[torch.Tensor] = None,
) -> Tuple[float, torch.Tensor]:
    cfg = load_config(config_path)

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

    audio = preprocess_audio(audio_path, audio_cfg)
    feats = compute_mfcc(audio, feat_cfg)
    frames = stack_frames(feats)
    features = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
    feat_lengths = torch.tensor([features.shape[1]], dtype=torch.long)

    if phoneme_segments is None:
        phoneme_segments = uniform_segments(features.shape[1], num_segments=10)
    phoneme_segments = phoneme_segments.unsqueeze(0)
    seg_lengths = torch.tensor([phoneme_segments.shape[1]], dtype=torch.long)

    model = build_model(cfg)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
    model.eval()

    with torch.no_grad():
        out = model(features, feat_lengths, phoneme_segments, seg_lengths)
        utterance_score = out["utterance_score"].item()
        phoneme_scores = out["phoneme_scores"].squeeze(0)

    return utterance_score, phoneme_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    score, phoneme_scores = infer(
        audio_path=args.audio,
        ref_text=args.text,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    print(f"Utterance score: {score:.2f}")
    print(f"Phoneme scores: {phoneme_scores.tolist()}")


if __name__ == "__main__":
    main()
