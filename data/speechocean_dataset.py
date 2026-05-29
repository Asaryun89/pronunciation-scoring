"""
Speechocean762 dataset for Fusion-C utterance-level pronunciation scoring.

Differences from data/dataset.py:
  • Score order matches the FusionScoringHead output convention:
        [total, accuracy, fluency, prosodic, completeness]
  • Transcripts are upper-cased and stripped of punctuation so they are
    clean for BGE tokenisation.
  • Scores are normalised to [0, 1]  (divide by SCORE_MAX = 10.0).
  • Provides fusion_collate_fn that returns transcripts as List[str]
    (BGE tokenises internally).

Standalone import:
    from data.speechocean_dataset import SpeechoceanFusionDataset, fusion_collate_fn
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

# Score field order expected by FusionScoringHead and finetune_fusionC trainer.
# Must stay in sync with FusionScoringHead docstring.
FUSION_SCORE_KEYS: List[str] = [
    "total",
    "accuracy",
    "fluency",
    "prosodic",
    "completeness",
]

# Mapping from our score keys to HuggingFace dataset field names.
_HF_FIELD: Dict[str, str] = {
    "total":        "score",
    "accuracy":     "accuracy",
    "fluency":      "fluency",
    "prosodic":     "prosodic",
    "completeness": "completeness",
}

SCORE_MAX: float   = 10.0
TARGET_SR: int     = 16_000

_PUNCT_RE = re.compile(r"[^\w\s]")   # strips everything except word chars and spaces


def _clean_transcript(text: str) -> str:
    """Uppercase and remove all punctuation."""
    return _PUNCT_RE.sub("", text).upper().strip()


class SpeechoceanFusionDataset(Dataset):
    """
    HuggingFace-backed Speechocean762 dataset for Fusion-C training.

    Each item returns:
        waveform:   Tensor [T]        — raw 16 kHz audio
        labels:     Tensor [5]        — normalised scores in [0, 1]
                                        order: [total, accuracy, fluency,
                                                prosodic, completeness]
        transcript: str               — uppercase, no punctuation
        utt_id:     str               — utterance identifier
        length:     Tensor scalar     — number of valid samples

    Args:
        cfg:          Full parsed config dict.
        split:        ``"train"`` or ``"test"``.
        augment:      Add small Gaussian noise to waveforms.
    """

    def __init__(
        self,
        cfg:     dict,
        split:   str  = "train",
        augment: bool = False,
    ) -> None:
        super().__init__()
        dcfg = cfg["data"]

        self.max_samples  = int(dcfg.get("max_duration_s", 20.0) * TARGET_SR)
        self.augment      = augment
        self._noise_amp   = 1e-4 if augment else 0.0
        self._tx_field    = dcfg.get("transcript_field", "text")

        try:
            from datasets import load_dataset, Audio
        except ImportError as exc:
            raise ImportError(
                "The `datasets` package is required.\n"
                "Install:  pip install datasets"
            ) from exc

        hf_id     = dcfg.get("hf_dataset_id", "mispeech/speechocean762")
        cache_dir = dcfg.get("hf_cache_dir", None)

        hf_ds = load_dataset(hf_id, split=split, cache_dir=cache_dir)
        hf_ds = hf_ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
        self._hf_ds = hf_ds

        # Build (N, 5) label tensor via column-wise access — avoids row-by-row Python loop.
        cols = np.column_stack([
            np.array(hf_ds[_HF_FIELD[k]], dtype=np.float32) / SCORE_MAX
            for k in FUSION_SCORE_KEYS
        ])
        self.labels: Tensor = torch.from_numpy(cols)   # (N, 5) in [0, 1]

        # Pre-clean all transcripts (small corpus — affordable up-front).
        raw_texts = hf_ds[self._tx_field]
        self._transcripts: List[str] = [
            _clean_transcript(t) for t in raw_texts
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Dataset interface
    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, idx: int) -> Dict[str, Tensor | str]:
        row   = self._hf_ds[idx]
        audio = row["audio"]   # {"array": ndarray, "sampling_rate": int, "path": str|None}

        waveform = torch.from_numpy(
            np.asarray(audio["array"], dtype=np.float32)
        )
        if waveform.dim() == 2:              # guard: mono-mix if stereo
            waveform = waveform.mean(0)
        waveform = waveform[: self.max_samples]

        if self._noise_amp > 0.0:
            waveform = waveform + torch.randn_like(waveform) * self._noise_amp

        audio_path = audio.get("path") or ""
        utt_id = Path(audio_path).stem if audio_path else f"utt_{idx:06d}"

        return {
            "utt_id":     utt_id,
            "waveform":   waveform,                                # (T,) float32
            "length":     torch.tensor(waveform.shape[0], dtype=torch.long),
            "labels":     self.labels[idx],                        # (5,) float32
            "transcript": self._transcripts[idx],
        }

    @property
    def score_keys(self) -> List[str]:
        return FUSION_SCORE_KEYS


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────

def fusion_collate_fn(
    batch: List[Dict[str, Tensor | str]],
) -> Dict[str, Tensor | List[str]]:
    """
    Collate a batch of SpeechoceanFusionDataset items.

    Returns:
        waveforms:      [B, T_max]   float32  — zero-padded
        attention_mask: [B, T_max]   long     — 1=real / 0=pad
        lengths:        [B]          long
        labels:         [B, 5]       float32  — normalised scores in [0, 1]
        transcripts:    List[str]    — BGE tokenises these internally
        utt_ids:        List[str]
    """
    utt_ids:     List[str]   = []
    transcripts: List[str]   = []
    waveforms:   List[Tensor] = []
    lengths:     List[Tensor] = []
    labels:      List[Tensor] = []

    for sample in batch:
        utt_ids.append(sample["utt_id"])
        transcripts.append(sample["transcript"])
        waveforms.append(sample["waveform"])
        lengths.append(sample["length"])
        labels.append(sample["labels"])

    max_len = max(w.shape[0] for w in waveforms)
    padded  = torch.stack(
        [F.pad(w, (0, max_len - w.shape[0])) for w in waveforms]
    )                                                              # [B, T_max]

    length_t = torch.stack(lengths)                               # [B]
    attn     = (torch.arange(max_len).unsqueeze(0) < length_t.unsqueeze(1)).long()

    return {
        "waveforms":      padded,
        "attention_mask": attn,
        "lengths":        length_t,
        "labels":         torch.stack(labels),                    # [B, 5]
        "transcripts":    transcripts,
        "utt_ids":        utt_ids,
    }
