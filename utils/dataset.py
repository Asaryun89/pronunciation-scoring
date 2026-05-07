from __future__ import annotations

"""
SpeechOcean762 dataset wrapper for pronunciation scoring.

Loads the ``mispeech/speechocean762`` dataset from HuggingFace, converts raw
audio and phoneme annotations into model-ready tensors, and provides a
``collate_fn`` for batched DataLoaders.

``librosa`` is imported lazily inside :func:`extract_prosody_features` so this
module can be imported even when librosa is not installed.
"""

from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from models.phoneme_vocab import PAD_ID, SIL_ID, phonemes_to_ids

# ─── Constants ────────────────────────────────────────────────────────────────

SCORE_KEYS: List[str] = ["accuracy", "completeness", "fluency", "prosody", "total"]
SCORE_MAX: float = 10.0

# ─── Score helpers ────────────────────────────────────────────────────────────


def normalize_scores(sample: dict) -> torch.Tensor:
    """Extract and normalise the 5 sentence-level scores to ``[0, 1]``.

    Args:
        sample: A dataset example containing sentence-level annotation keys.

    Returns:
        ``(5,)`` float32 tensor of scores divided by :data:`SCORE_MAX`.
    """
    scores = [float(sample[k]) for k in SCORE_KEYS]
    return torch.tensor(scores, dtype=torch.float32) / SCORE_MAX


# ─── Phoneme sequence builder ─────────────────────────────────────────────────


def build_phoneme_sequence(words: List[Any], insert_silence: bool = True) -> List[int]:
    """Convert a list of word annotation dicts into a flat phoneme ID list.

    Args:
        words: List of word dicts, each containing a ``"phones"`` key with a
            list of raw phoneme token strings.
        insert_silence: When ``True``, inserts :data:`~models.phoneme_vocab.SIL_ID`
            between consecutive words (not before the first word).

    Returns:
        Flat list of integer phoneme IDs.
    """
    ids: List[int] = []
    for i, word in enumerate(words):
        phones: List[Any] = word.get("phones", []) or []
        if not phones:
            continue
        if insert_silence and i > 0:
            ids.append(SIL_ID)
        ids.extend(phonemes_to_ids(phones))
    return ids


# ─── Prosody feature extraction ───────────────────────────────────────────────


def extract_prosody_features(audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
    """Compute 5 hand-crafted prosodic features from a raw waveform.

    Features are returned in fixed order:
    ``[rms, zcr, peak_rate, pitch_mean, pitch_std]``

    Args:
        audio: 1-D float32 numpy array of audio samples.
        sr: Sample rate in Hz.

    Returns:
        ``(5,)`` float32 tensor.  Any NaN or infinite values are replaced with 0.
    """
    import librosa  # lazy import — module usable without librosa installed

    if audio.size == 0:
        return torch.zeros(5, dtype=torch.float32)

    duration = audio.size / sr

    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)[0]))
    peak_rate = float(
        np.sum(np.abs(audio) > 0.3 * np.max(np.abs(audio)) + 1e-8) / (duration + 1e-12)
    )

    try:
        f0 = librosa.yin(audio, fmin=50.0, fmax=400.0, sr=sr)
        finite = f0[np.isfinite(f0)]
        pitch_mean = float(np.mean(finite)) if finite.size > 0 else 0.0
        pitch_std = float(np.std(finite)) if finite.size > 0 else 0.0
    except Exception:
        pitch_mean = 0.0
        pitch_std = 0.0

    feats = torch.tensor(
        [rms, zcr, peak_rate, pitch_mean, pitch_std], dtype=torch.float32
    )
    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# ─── Dataset class ────────────────────────────────────────────────────────────


class SpeechOcean762Dataset(Dataset):
    """Torch Dataset wrapping the SpeechOcean762 HuggingFace dataset.

    Each item returns a dict of tensors ready for the pronunciation scoring
    model.

    Args:
        split: Dataset split — typically ``"train"`` or ``"test"``.
        max_audio_seconds: Truncate audio exceeding this duration (seconds).
        insert_silence: Insert silence tokens between words in phoneme sequences.
    """

    def __init__(
        self,
        split: str = "train",
        max_audio_seconds: float = 20.0,
        insert_silence: bool = True,
    ) -> None:
        from datasets import load_dataset  # lazy import

        self.data = load_dataset("mispeech/speechocean762", split=split)
        self.max_audio_seconds = max_audio_seconds
        self.insert_silence = insert_silence

    def __len__(self) -> int:
        """Return the number of utterances in the split."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return tensors for one utterance.

        Returns:
            Dict with keys:

            - ``"input_values"``: ``(T,)`` float32 audio waveform.
            - ``"phoneme_ids"``: ``(P,)`` long phoneme ID sequence.
            - ``"prosody_feats"``: ``(5,)`` float32 prosodic features.
            - ``"sent_scores"``: ``(5,)`` float32 normalised scores in ``[0, 1]``.
        """
        sample = self.data[idx]

        audio_dict = sample["audio"]
        audio: np.ndarray = np.array(audio_dict["array"], dtype=np.float32)
        sr: int = int(audio_dict["sampling_rate"])
        max_samples = int(self.max_audio_seconds * sr)
        audio = audio[:max_samples]

        words = sample.get("words", []) or []
        phoneme_ids = build_phoneme_sequence(words, self.insert_silence)
        if not phoneme_ids:
            phoneme_ids = [SIL_ID]

        return {
            "input_values": torch.tensor(audio, dtype=torch.float32),
            "phoneme_ids": torch.tensor(phoneme_ids, dtype=torch.long),
            "prosody_feats": extract_prosody_features(audio, sr),
            "sent_scores": normalize_scores(sample),
        }


# ─── Collate function ─────────────────────────────────────────────────────────


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-length samples into stacked tensors.

    Args:
        batch: List of dicts returned by :meth:`SpeechOcean762Dataset.__getitem__`.

    Returns:
        Dict with padded batch tensors:

        - ``"input_values"``: ``(B, T_max)`` float32, zero-padded.
        - ``"attention_mask"``: ``(B, T_max)`` long, 1 = valid, 0 = pad.
        - ``"phoneme_ids"``: ``(B, P_max)`` long, padded with :data:`~models.phoneme_vocab.PAD_ID`.
        - ``"phoneme_mask"``: ``(B, P_max)`` bool, ``True`` = valid token.
        - ``"prosody_feats"``: ``(B, 5)`` float32 stacked prosody features.
        - ``"sent_scores"``: ``(B, 5)`` float32 stacked normalised scores.
    """
    max_audio = max(s["input_values"].size(0) for s in batch)
    max_phone = max(s["phoneme_ids"].size(0) for s in batch)
    B = len(batch)

    input_values = torch.zeros(B, max_audio, dtype=torch.float32)
    attention_mask = torch.zeros(B, max_audio, dtype=torch.long)
    phoneme_ids = torch.full((B, max_phone), PAD_ID, dtype=torch.long)
    phoneme_mask = torch.zeros(B, max_phone, dtype=torch.bool)

    for i, sample in enumerate(batch):
        T = sample["input_values"].size(0)
        P = sample["phoneme_ids"].size(0)
        input_values[i, :T] = sample["input_values"]
        attention_mask[i, :T] = 1
        phoneme_ids[i, :P] = sample["phoneme_ids"]
        phoneme_mask[i, :P] = True

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "phoneme_ids": phoneme_ids,
        "phoneme_mask": phoneme_mask,
        "prosody_feats": torch.stack([s["prosody_feats"] for s in batch]),
        "sent_scores": torch.stack([s["sent_scores"] for s in batch]),
    }
