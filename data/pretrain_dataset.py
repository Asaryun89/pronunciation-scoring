"""
Dataset and collation for HuBERT-style self-supervised pre-training.

Audio is streamed from the HuggingFace Hub (mispeech/speechocean762).
Cluster-label targets are loaded from a pre-computed pickle produced by
``training.kmeans.KMeansQuantizer.assign_and_save()``.

Each sample provides:
  waveform    (T,)          raw 16 kHz audio
  hi_targets  (T_feat,)     cluster IDs for the high-res unit-prediction head
  lo_targets  (T_feat',)    cluster IDs for the low-res unit-prediction head
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

_TARGET_SR     = 16_000
_HUBERT_STRIDE = 320


class PretrainDataset(Dataset):
    """
    PyTorch Dataset for HuBERT pre-training backed by the HuggingFace Hub.

    Args:
        labels_path:    Path to the pickle produced by KMeansQuantizer.assign_and_save().
        hf_dataset_id:  HuggingFace dataset identifier.
        split:          Dataset split ("train" or "test").
        max_duration_s: Audio longer than this (seconds) is truncated.
        cnn_stride:     CNN temporal stride (320 for HuBERT base).
        cache_dir:      HF cache directory (None = HF default).
    """

    def __init__(
        self,
        labels_path:   str,
        hf_dataset_id: str           = "mispeech/speechocean762",
        split:         str           = "train",
        max_duration_s: float        = 20.0,
        cnn_stride:    int           = _HUBERT_STRIDE,
        cache_dir:     Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_samples = int(max_duration_s * _TARGET_SR)
        self.cnn_stride  = cnn_stride

        try:
            from datasets import load_dataset, Audio
        except ImportError as exc:
            raise ImportError(
                "The `datasets` package is required.\n"
                "Install it with:  pip install datasets"
            ) from exc

        # Load raw dataset, then cast with decode=False to read path metadata
        # without decoding audio.  Audio(decode=False) always yields plain dicts
        # {"path": ..., "bytes": ...} — the same format KMeansQuantizer uses when
        # iterating decoded rows, so utt_ids are guaranteed to match.
        raw_ds  = load_dataset(hf_dataset_id, split=split, cache_dir=cache_dir)
        path_ds = raw_ds.cast_column("audio", Audio(sampling_rate=_TARGET_SR, decode=False))
        audio_meta: list = path_ds["audio"]   # [{"path": str|None, "bytes": bytes}, ...]

        all_utt_ids: List[str] = [
            Path(
                (a.get("path") or "") if isinstance(a, dict)
                else (getattr(a, "path", "") or "")
            ).stem or f"utt_{i:06d}"
            for i, a in enumerate(audio_meta)
        ]

        # Load pre-computed cluster labels.
        with open(labels_path, "rb") as f:
            all_labels: Dict[str, Dict[str, np.ndarray]] = pickle.load(f)

        # Filter to rows that have labels, then cast audio to 16 kHz.
        valid: List[Tuple[int, str]] = [
            (i, uid) for i, uid in enumerate(all_utt_ids) if uid in all_labels
        ]
        if not valid:
            sample_ds  = all_utt_ids[:5]
            sample_lbl = list(all_labels.keys())[:5]
            raise RuntimeError(
                f"Zero utterances matched between the dataset and the labels file.\n"
                f"  Dataset utt_ids (first 5): {sample_ds}\n"
                f"  Labels file keys (first 5): {sample_lbl}\n"
                f"  Labels file: {labels_path}\n"
                f"The labels were likely generated in a different environment. "
                f"Re-run k-means:\n"
                f"  python pretrain.py --config <your_config> --prepare-kmeans"
            )

        missing = len(raw_ds) - len(valid)
        if missing:
            log.warning("%d utterances missing from labels file — skipped.", missing)

        valid_indices = [i for i, _ in valid]
        self._utt_ids: List[str] = [uid for _, uid in valid]

        hf_ds = raw_ds.cast_column("audio", Audio(sampling_rate=_TARGET_SR))
        # select() returns a view — no data is copied.
        self._hf_ds = hf_ds.select(valid_indices) if len(valid_indices) < len(hf_ds) else hf_ds
        self._labels: Dict[str, Dict[str, np.ndarray]] = {
            uid: all_labels[uid] for uid in self._utt_ids
        }

    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._utt_ids)

    def __getitem__(self, idx: int) -> Dict[str, Tensor | str]:
        row    = self._hf_ds[idx]
        utt_id = self._utt_ids[idx]

        arr      = np.asarray(row["audio"]["array"], dtype=np.float32)
        waveform = torch.from_numpy(arr)
        if waveform.dim() == 2:
            waveform = waveform.mean(0)
        waveform = waveform[: self.max_samples]

        feat_len = waveform.shape[0] // self.cnn_stride

        hi_labels = self._labels[utt_id]["hi"]   # (T_feat_full,) np.int16
        lo_labels = self._labels[utt_id]["lo"]   # (T_feat_full // stride,)

        hi_targets = torch.from_numpy(hi_labels[:feat_len].astype(np.int64))
        lo_targets = torch.from_numpy(lo_labels[: feat_len // 2].astype(np.int64))

        return {
            "utt_id":     utt_id,
            "waveform":   waveform,
            "length":     torch.tensor(waveform.shape[0], dtype=torch.long),
            "hi_targets": hi_targets,
            "lo_targets": lo_targets,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_collate_fn(
    batch: List[Dict[str, Tensor | str]],
) -> Dict[str, Tensor | List[str]]:
    """
    Pad waveforms to the longest in the batch; pad targets with -1
    (the ``ignore_index`` for ``nn.CrossEntropyLoss``).

    Returns keys:
      waveforms        (B, T_max)    float32
      attention_mask   (B, T_max)    long    — 1=real, 0=pad
      lengths          (B,)          long
      hi_targets       (B, F_max)    long    — -1=pad
      lo_targets       (B, F'_max)   long    — -1=pad
      utt_ids          List[str]
    """
    utt_ids:    List[str]    = []
    waveforms:  List[Tensor] = []
    lengths:    List[Tensor] = []
    hi_targets: List[Tensor] = []
    lo_targets: List[Tensor] = []

    for s in batch:
        utt_ids.append(s["utt_id"])
        waveforms.append(s["waveform"])
        lengths.append(s["length"])
        hi_targets.append(s["hi_targets"])
        lo_targets.append(s["lo_targets"])

    max_wav = max(w.shape[0] for w in waveforms)
    max_hi  = max(t.shape[0] for t in hi_targets)
    max_lo  = max(t.shape[0] for t in lo_targets)

    padded_wav = torch.stack(
        [F.pad(w, (0, max_wav - w.shape[0])) for w in waveforms]
    )
    length_t  = torch.stack(lengths)
    attn_mask = (
        torch.arange(max_wav).unsqueeze(0) < length_t.unsqueeze(1)
    ).long()
    padded_hi = torch.stack(
        [F.pad(t, (0, max_hi - t.shape[0]), value=-1) for t in hi_targets]
    )
    padded_lo = torch.stack(
        [F.pad(t, (0, max_lo - t.shape[0]), value=-1) for t in lo_targets]
    )

    return {
        "waveforms":      padded_wav,
        "attention_mask": attn_mask,
        "lengths":        length_t,
        "hi_targets":     padded_hi,
        "lo_targets":     padded_lo,
        "utt_ids":        utt_ids,
    }
