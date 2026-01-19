from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from features.mfcc import MFCCConfig, compute_mfcc, stack_frames
from features.pitch import PitchConfig, compute_pitch
from utils.alignment import dtw_alignment, uniform_alignment, uniform_frame_alignment
from utils.audio import AudioConfig, preprocess_audio


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def distance_to_score(distance: float, scale: float) -> float:
    """Convert a distance to a 0-100 score where smaller distance is higher score."""
    score = 100.0 * np.exp(-distance / max(scale, 1e-6))
    return float(np.clip(score, 0.0, 100.0))


def _trim_feature_pair(mfcc: np.ndarray, pitch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    frames = min(mfcc.shape[0], pitch.shape[0])
    return mfcc[:frames], pitch[:frames]


def _map_frames_to_words(num_frames: int, words: List[str]) -> List[int]:
    lengths = uniform_alignment(num_frames, len(words))
    frame_to_word = [0] * num_frames
    offset = 0
    for idx, length in enumerate(lengths):
        end = min(offset + length, num_frames)
        for i in range(offset, end):
            frame_to_word[i] = idx
        offset = end
    return frame_to_word


def infer(
    learner_audio_path: str,
    config_path: str,
    reference_text: Optional[str],
    reference_audio_path: str,
    alignment_method: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, Any]]:
    cfg = load_config(config_path)

    audio_cfg = AudioConfig(
        sample_rate=cfg["sample_rate"],
        max_audio_seconds=cfg["max_audio_seconds"],
    )
    mfcc_cfg = MFCCConfig(
        sample_rate=cfg["sample_rate"],
        n_mfcc=cfg["feature"].get("n_mfcc", 13),
        n_fft=cfg["feature"]["n_fft"],
        hop_length=cfg["feature"]["hop_length"],
        win_length=cfg["feature"]["win_length"],
        add_deltas=cfg["feature"].get("add_deltas", True),
    )
    pitch_cfg = PitchConfig(
        sample_rate=cfg["sample_rate"],
        hop_length=cfg["feature"]["hop_length"],
        win_length=cfg["feature"]["win_length"],
        fmin=cfg.get("pitch", {}).get("fmin", 50.0),
        fmax=cfg.get("pitch", {}).get("fmax", 500.0),
    )
    alignment_method = alignment_method or cfg.get("alignment", {}).get("method", "dtw")
    pitch_weight = cfg.get("scoring", {}).get("pitch_weight", 0.01)
    distance_scale = cfg.get("scoring", {}).get("distance_scale", 10.0)

    learner_audio = preprocess_audio(learner_audio_path, audio_cfg)
    reference_audio = preprocess_audio(reference_audio_path, audio_cfg)

    ref_mfcc = stack_frames(compute_mfcc(reference_audio, mfcc_cfg))
    learner_mfcc = stack_frames(compute_mfcc(learner_audio, mfcc_cfg))
    ref_pitch = compute_pitch(reference_audio, pitch_cfg)
    learner_pitch = compute_pitch(learner_audio, pitch_cfg)
    ref_mfcc, ref_pitch = _trim_feature_pair(ref_mfcc, ref_pitch)
    learner_mfcc, learner_pitch = _trim_feature_pair(learner_mfcc, learner_pitch)

    if alignment_method == "dtw":
        path = dtw_alignment(ref_mfcc, learner_mfcc)
    elif alignment_method == "forced":
        path = uniform_frame_alignment(ref_mfcc.shape[0], learner_mfcc.shape[0])
    else:
        raise ValueError(f"Unknown alignment method: {alignment_method}")

    frame_scores: List[float] = []
    mfcc_distances: List[float] = []
    pitch_diffs: List[float] = []
    for ref_idx, learner_idx in path:
        mfcc_dist = float(np.linalg.norm(ref_mfcc[ref_idx] - learner_mfcc[learner_idx]))
        pitch_diff = float(abs(ref_pitch[ref_idx] - learner_pitch[learner_idx]))
        combined = mfcc_dist + (pitch_weight * pitch_diff)
        frame_scores.append(distance_to_score(combined, distance_scale))
        mfcc_distances.append(mfcc_dist)
        pitch_diffs.append(pitch_diff)

    words: List[str] = []
    if reference_text:
        words = [w for w in reference_text.strip().split() if w]
    if not words:
        words = ["<segment>"]
    frame_to_word = _map_frames_to_words(ref_mfcc.shape[0], words)
    word_scores: List[float] = [0.0 for _ in words]
    word_counts: List[int] = [0 for _ in words]
    for (ref_idx, _), score in zip(path, frame_scores):
        word_idx = frame_to_word[min(ref_idx, len(frame_to_word) - 1)]
        word_scores[word_idx] += score
        word_counts[word_idx] += 1
    for i, count in enumerate(word_counts):
        if count > 0:
            word_scores[i] /= count
        else:
            word_scores[i] = 0.0

    sentence_score = float(np.mean(word_scores)) if word_scores else 0.0
    scores = {
        "overall": sentence_score,
        "avg_mfcc_distance": float(np.mean(mfcc_distances)) if mfcc_distances else 0.0,
        "avg_pitch_diff_hz": float(np.mean(pitch_diffs)) if pitch_diffs else 0.0,
    }
    feedback = score_to_feedback(sentence_score)
    details = {
        "words": words,
        "word_scores": word_scores,
        "alignment": alignment_method,
        "frame_count": len(frame_scores),
        "has_reference_text": bool(reference_text),
    }
    return scores, feedback, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-audio", type=str, required=True)
    parser.add_argument("--reference-audio", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--alignment", type=str, default=None, choices=["dtw", "forced"])
    parser.add_argument("--feedback-out", type=str, default=None)
    args = parser.parse_args()

    scores, feedback, details = infer(
        learner_audio_path=args.learner_audio,
        config_path=args.config,
        reference_text=args.text,
        reference_audio_path=args.reference_audio,
        alignment_method=args.alignment,
    )
    print(f"Utterance score: {scores['overall']:.2f}")
    print(f"Avg MFCC distance: {scores['avg_mfcc_distance']:.2f}")
    print(f"Avg pitch diff (Hz): {scores['avg_pitch_diff_hz']:.2f}")
    print(f"Feedback: {feedback['message']} (band={feedback['band']})")

    if args.feedback_out:
        payload = {"scores": scores, **feedback, **details}
        out_path = Path(args.feedback_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
