"""
Speechocean762 dataset loader — backed by the HuggingFace Hub.

Dataset: mispeech/speechocean762
  https://huggingface.co/datasets/mispeech/speechocean762

HF field layout
  audio         — Audio feature (dict: array, sampling_rate, path)
  text          — utterance transcript
  speaker_id    — speaker label
  accuracy      — 0–10 score
  fluency       — 0–10 score
  completeness  — 0–10 score
  prosodic      — 0–10 score
  score         — 0–10 total score
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

SCORE_KEYS: List[str] = [
    "accuracy",
    "fluency",
    "completeness",
    "prosodic",
    "total",
]
SCORE_MAX = 10.0
TARGET_SAMPLE_RATE = 16_000

# Maps our SCORE_KEYS names to the field names used in the HF dataset.
_HF_FIELD: Dict[str, str] = {
    "accuracy":    "accuracy",
    "fluency":     "fluency",
    "completeness":"completeness",
    "prosodic":    "prosodic",
    "total":       "score",
}


class Speechocean762Dataset(Dataset):
    """
    PyTorch Dataset for Speechocean762 loaded from the HuggingFace Hub.

    Args:
        split:          "train" or "test".
        hf_dataset_id:  HuggingFace dataset identifier.
        max_duration_s: Waveforms longer than this (seconds) are truncated.
        augment:        Apply Gaussian noise augmentation during training.
        cache_dir:      HF cache directory (None = HF default).
    """

    def __init__(
        self,
        split: str = "train",
        hf_dataset_id: str = "mispeech/speechocean762",
        max_duration_s: float = 30.0,
        augment: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_samples = int(max_duration_s * TARGET_SAMPLE_RATE)
        self.augment = augment
        self._noise_amp = 1e-4 if augment else 0.0

        try:
            from datasets import load_dataset, Audio
        except ImportError as exc:
            raise ImportError(
                "The `datasets` package is required.\n"
                "Install it with:  pip install datasets"
            ) from exc

        hf_ds = load_dataset(hf_dataset_id, split=split, cache_dir=cache_dir)
        # Decode + resample to 16 kHz on-the-fly when each row is accessed.
        hf_ds = hf_ds.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
        self._hf_ds = hf_ds

        # Build (N, 5) label tensor using column-wise access — much faster than
        # iterating row by row for a ~5 k-sample corpus.
        cols = np.column_stack([
            np.array(hf_ds[_HF_FIELD[k]], dtype=np.float32) / SCORE_MAX
            for k in SCORE_KEYS
        ])
        self.labels: Tensor = torch.from_numpy(cols)

    # ──────────────────────────────────────────────────────────────────────
    # Dataset interface
    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, idx: int) -> Dict[str, Tensor | str]:
        row = self._hf_ds[idx]

        # audio is a dict {"array": np.ndarray, "sampling_rate": int, "path": str|None}
        audio = row["audio"]
        waveform = torch.from_numpy(
            np.asarray(audio["array"], dtype=np.float32)
        )

        if waveform.dim() == 2:          # guard: mix to mono if stereo
            waveform = waveform.mean(0)

        waveform = waveform[: self.max_samples]

        if self._noise_amp > 0.0:
            waveform = waveform + torch.randn_like(waveform) * self._noise_amp

        audio_path = audio.get("path") or ""
        utt_id = Path(audio_path).stem if audio_path else f"utt_{idx:06d}"

        return {
            "utt_id":     utt_id,
            "waveform":   waveform,                                    # (T,) float32
            "length":     torch.tensor(waveform.shape[0], dtype=torch.long),
            "labels":     self.labels[idx],                            # (5,) float32
            "transcript": row.get("text", ""),
            "speaker":    str(row.get("speaker_id", "unknown")),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    @property
    def score_keys(self) -> List[str]:
        return SCORE_KEYS

    def get_label_stats(self) -> pd.DataFrame:
        """Return per-dimension mean / std over the full split."""
        arr = self.labels.numpy()
        return pd.DataFrame(
            {"mean": arr.mean(0), "std": arr.std(0)},
            index=SCORE_KEYS,
        )
