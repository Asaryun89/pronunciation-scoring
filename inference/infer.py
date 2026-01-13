from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

from features.logmel import LogMelConfig, compute_log_mel, stack_frames
from models.pronunciation_model import ModelConfig, PronunciationModel
from utils.alignment import uniform_alignment
from utils.audio import AudioConfig, preprocess_audio
from utils.g2p import text_to_phonemes


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


def score_to_feedback(score: float) -> Dict[str, str]:
    """Map a numeric score to a simple qualitative feedback band."""
    if score >= 85:
        band = "excellent"
        message = "Great pronunciation clarity and consistency."
    elif score >= 70:
        band = "good"
        message = "Solid pronunciation with minor issues to refine."
    elif score >= 50:
        band = "fair"
        message = "Understandable but needs more consistency and practice."
    else:
        band = "needs_practice"
        message = "Pronunciation needs focused practice for improvement."
    return {"band": band, "message": message}


def infer(
    audio_path: str,
    config_path: str,
    reference_text: str,
    checkpoint_path: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, Any]]:
    cfg = load_config(config_path)

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

    audio = preprocess_audio(audio_path, audio_cfg)
    feats = compute_log_mel(audio, feat_cfg)
    frames = stack_frames(feats)
    features = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
    feat_lengths = torch.tensor([features.shape[1]], dtype=torch.long)

    phoneme_symbols, phoneme_ids = text_to_phonemes(reference_text)
    phoneme_lengths = uniform_alignment(int(feat_lengths.item()), len(phoneme_ids))
    phoneme_lengths_tensor = torch.tensor([phoneme_lengths], dtype=torch.long)

    model = build_model(cfg)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
    model.eval()

    with torch.no_grad():
        out = model(features, feat_lengths, phoneme_lengths_tensor)

    utterance_score = out["utterance_score"].item()
    scores = {
        "overall": utterance_score,
        "accuracy": out["accuracy_score"].item(),
        "fluency": out["fluency_score"].item(),
        "prosody": out["prosody_score"].item(),
    }
    feedback = score_to_feedback(utterance_score)

    phoneme_scores = out["phoneme_scores"].squeeze(0).tolist()
    per_phoneme = [
        {"phoneme": sym, "score": float(score)}
        for sym, score in zip(phoneme_symbols, phoneme_scores)
    ]
    details = {
        "phonemes": phoneme_symbols,
        "phoneme_ids": phoneme_ids,
        "per_phoneme": per_phoneme,
    }
    return scores, feedback, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--feedback-out", type=str, default=None)
    args = parser.parse_args()

    scores, feedback, details = infer(
        audio_path=args.audio,
        config_path=args.config,
        reference_text=args.text,
        checkpoint_path=args.checkpoint,
    )
    print(f"Utterance score: {scores['overall']:.2f}")
    print(
        "Breakdown: "
        f"accuracy={scores['accuracy']:.2f}, "
        f"fluency={scores['fluency']:.2f}, "
        f"prosody={scores['prosody']:.2f}"
    )
    print(f"Feedback: {feedback['message']} (band={feedback['band']})")

    if args.feedback_out:
        payload = {"scores": scores, **feedback, **details}
        out_path = Path(args.feedback_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
