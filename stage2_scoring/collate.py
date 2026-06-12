# Batch collator for SpeechOcean762 training (stage2_scoring/train.py).
#
# Produces the batch keys consumed by HubertMultiTask.forward() and compute_loss():
#   input_values        (B, T_samples)  float32 — zero-mean/unit-std waveforms, zero-padded
#   attention_mask      (B, T_samples)  long    — 1 = real sample, 0 = padding
#   sent_targets        (B, 5)          float32 — SENT_DIMS order, scaled to [0, 1]
#   prosody_feats       (B, 5)          float32 — PROSODY_DIMS order, see utils/prosody.py
#   text_input_ids      (B, L)          long    — only when a text tokenizer is provided
#   text_attention_mask (B, L)          long    — only when a text tokenizer is provided
#
# Audio decoding expects rows cast with `Audio(decode=False)` (see train.py), i.e.
# example["audio"] = {"path": str, "bytes": bytes}. Preprocessing mirrors the
# inference path in inference/predictor.py: resample 16 kHz → peak normalize →
# VAD trim → peak normalize → zero-mean/unit-std.

import io
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from stage2_scoring.constants import SENT_DIMS, SENT_SCALE                       # noqa: E402
from utils.preprocessing import resample, peak_normalize, vad_trim       # noqa: E402
from utils.prosody import basic_prosody_features, prosody_to_vector      # noqa: E402

# HuBERT's CNN feature extractor needs a minimum number of samples to emit
# one frame; pad anything shorter (e.g. an over-aggressive VAD trim) up to this.
_MIN_SAMPLES = 640


class Collator:
    """
    Collate SpeechOcean762 rows into model-ready batches.

    Parameters
    ----------
    sample_rate     : target sample rate for HuBERT input (16 kHz)
    text_tokenizer  : HuggingFace tokenizer for the text stream (None = disabled)
    split_name      : split label used in error messages only
    use_vad         : apply webrtcvad trimming (matches inference preprocessing)
    text_max_length : truncation length for reference-text tokenization
    """

    def __init__(
        self,
        sample_rate:     int           = 16000,
        text_tokenizer:  Optional[Any] = None,
        split_name:      str           = "train",
        use_vad:         bool          = True,
        text_max_length: int           = 128,
    ):
        self.sample_rate     = sample_rate
        self.text_tokenizer  = text_tokenizer
        self.split_name      = split_name
        self.use_vad         = use_vad
        self.text_max_length = text_max_length

    # ------------------------------------------------------------------
    def _decode_audio(self, example: Dict[str, Any], idx: int) -> np.ndarray:
        audio_info = example.get("audio")
        if not isinstance(audio_info, dict):
            raise ValueError(
                f"{self.split_name}[{idx}]: expected dict 'audio' column "
                f"(cast with Audio(decode=False)), got {type(audio_info).__name__}"
            )

        raw = audio_info.get("bytes")
        if raw:
            data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        elif audio_info.get("path"):
            data, sr = sf.read(audio_info["path"], dtype="float32")
        else:
            raise ValueError(
                f"{self.split_name}[{idx}]: audio column has neither 'bytes' nor 'path'"
            )

        if data.ndim > 1:
            data = data.mean(axis=1)
        data, sr = resample(data.astype(np.float32), sr, self.sample_rate)

        # Same chain as utils.preprocessing.preprocess_wav (inference parity)
        data = peak_normalize(data)
        if self.use_vad:
            data = vad_trim(data, sr=sr)
            data = peak_normalize(data)

        if len(data) < _MIN_SAMPLES:
            data = np.pad(data, (0, _MIN_SAMPLES - len(data)))
        return data

    # ------------------------------------------------------------------
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [self._decode_audio(ex, i) for i, ex in enumerate(examples)]

        # Prosody targets are computed on the peak-normalized audio, before
        # zero-mean/unit-std — matching inference/predictor.py.
        prosody_feats = torch.from_numpy(np.stack([
            prosody_to_vector(basic_prosody_features(a, sr=self.sample_rate))
            for a in audios
        ]))

        # Zero-mean/unit-std per utterance, then zero-pad to the batch max length.
        batch_size = len(audios)
        max_len    = max(len(a) for a in audios)
        input_values   = torch.zeros(batch_size, max_len, dtype=torch.float32)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, a in enumerate(audios):
            norm = (a - a.mean()) / (a.std() + 1e-7)
            input_values[i, : len(a)]   = torch.from_numpy(norm)
            attention_mask[i, : len(a)] = 1

        # Utterance labels → [0, 1] in SENT_DIMS order.
        sent_targets = torch.tensor(
            [[float(ex[dim]) / SENT_SCALE for dim in SENT_DIMS] for ex in examples],
            dtype=torch.float32,
        )

        batch: Dict[str, torch.Tensor] = {
            "input_values":   input_values,
            "attention_mask": attention_mask,
            "sent_targets":   sent_targets,
            "prosody_feats":  prosody_feats,
        }

        if self.text_tokenizer is not None:
            text_enc = self.text_tokenizer(
                [ex["text"] for ex in examples],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.text_max_length,
            )
            batch["text_input_ids"]      = text_enc["input_ids"]
            batch["text_attention_mask"] = text_enc["attention_mask"]

        return batch
