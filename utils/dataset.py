from __future__ import annotations

"""
SpeechOcean762 dataset wrapper for pronunciation scoring with BGE text embeddings.

Loads the ``mispeech/speechocean762`` dataset from HuggingFace, tokenises the
script text for BGE, extracts prosody features from audio, and provides a
``collate_fn`` for batched DataLoaders.

``librosa`` is imported lazily inside :func:`extract_prosody_features` so this
module can be imported even when librosa is not installed.
"""

from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# ─── Constants ────────────────────────────────────────────────────────────────

SCORE_KEYS: List[str] = ["accuracy", "completeness", "fluency", "prosody", "total"]
SCORE_MAX: float = 10.0
BGE_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
BGE_MAX_LENGTH: int = 128  # scripts are short


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
        bge_model_name: HuggingFace model ID for the BGE tokenizer.
        max_text_length: Maximum token length for script tokenisation.
    """

    def __init__(
        self,
        split: str = "train",
        max_audio_seconds: float = 20.0,
        bge_model_name: str = BGE_MODEL_NAME,
        max_text_length: int = BGE_MAX_LENGTH,
    ) -> None:
        from datasets import load_dataset  # lazy import

        self.data = load_dataset("mispeech/speechocean762", split=split)
        self.max_audio_seconds = max_audio_seconds
        self.max_text_length = max_text_length
        self.tokenizer = AutoTokenizer.from_pretrained(bge_model_name)

    def __len__(self) -> int:
        """Return the number of utterances in the split."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return tensors for one utterance.

        Returns:
            Dict with keys:

            - ``"input_values"``: ``(T,)`` float32 audio waveform.
            - ``"text_input_ids"``: ``(L,)`` long tokenised script.
            - ``"text_attention_mask"``: ``(L,)`` long 1=valid, 0=pad.
            - ``"prosody_feats"``: ``(5,)`` float32 prosodic features.
            - ``"sent_scores"``: ``(5,)`` float32 normalised scores in ``[0, 1]``.
        """
        sample = self.data[idx]

        audio_dict = sample["audio"]
        audio: np.ndarray = np.array(audio_dict["array"], dtype=np.float32)
        sr: int = int(audio_dict["sampling_rate"])
        max_samples = int(self.max_audio_seconds * sr)
        audio = audio[:max_samples]

        encoded = self.tokenizer(
            sample["text"],
            max_length=self.max_text_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = encoded["input_ids"].squeeze(0)         # (L,)
        text_attention_mask = encoded["attention_mask"].squeeze(0)  # (L,)

        try:
            prosody_feats = extract_prosody_features(audio, sr)
        except Exception:
            prosody_feats = torch.zeros(5, dtype=torch.float32)

        return {
            "input_values": torch.tensor(audio, dtype=torch.float32),
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "prosody_feats": prosody_feats,
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
        - ``"audio_attention_mask"``: ``(B, T_max)`` long, 1=valid, 0=pad.
        - ``"text_input_ids"``: ``(B, L_max)`` long, padded with tokenizer pad_token_id.
        - ``"text_attention_mask"``: ``(B, L_max)`` long, 1=valid, 0=pad.
        - ``"prosody_feats"``: ``(B, 5)`` float32 stacked prosody features.
        - ``"sent_scores"``: ``(B, 5)`` float32 stacked normalised scores.
    """
    # Infer pad_token_id from the first item's attention mask length — stored on dataset
    # We use 0 as a safe default; the tokenizer's pad_token_id is passed via closure
    # below via _get_pad_token_id. Instead we derive it from the batch structure.
    # BGE tokenizer pad_token_id is 0 ([PAD] token), but we rely on what's stored.
    # Since collate_fn doesn't have dataset reference, we use a module-level default.
    _pad_token_id = 0  # BGE/BERT-style tokenizers use 0 for [PAD]

    max_audio = max(s["input_values"].size(0) for s in batch)
    max_text = max(s["text_input_ids"].size(0) for s in batch)
    B = len(batch)

    input_values = torch.zeros(B, max_audio, dtype=torch.float32)
    audio_attention_mask = torch.zeros(B, max_audio, dtype=torch.long)
    text_input_ids = torch.full((B, max_text), _pad_token_id, dtype=torch.long)
    text_attention_mask = torch.zeros(B, max_text, dtype=torch.long)

    for i, sample in enumerate(batch):
        T = sample["input_values"].size(0)
        L = sample["text_input_ids"].size(0)
        input_values[i, :T] = sample["input_values"]
        audio_attention_mask[i, :T] = 1
        text_input_ids[i, :L] = sample["text_input_ids"]
        text_attention_mask[i, :L] = sample["text_attention_mask"]

    return {
        "input_values": input_values,
        "audio_attention_mask": audio_attention_mask,
        "text_input_ids": text_input_ids,
        "text_attention_mask": text_attention_mask,
        "prosody_feats": torch.stack([s["prosody_feats"] for s in batch]),
        "sent_scores": torch.stack([s["sent_scores"] for s in batch]),
    }
