"""
Collation utility for variable-length waveforms.

Pads each batch to the longest sequence in that batch and produces an
attention mask (1 = real sample, 0 = padding).
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor


def collate_fn(
    batch: List[Dict[str, Tensor | str]],
) -> Dict[str, Tensor | List[str]]:
    """
    Args:
        batch: list of dicts returned by Speechocean762Dataset.__getitem__.

    Returns:
        Batched dict with keys:
          waveforms   (B, T_max)   float32  — zero-padded
          attention_mask (B, T_max) long    — 1 real / 0 pad
          lengths     (B,)         long     — original sample counts
          labels      (B, 5)       float32
          utt_ids     List[str]
          transcripts List[str]
          speakers    List[str]
    """
    utt_ids: List[str] = []
    transcripts: List[str] = []
    speakers: List[str] = []
    waveforms: List[Tensor] = []
    lengths: List[Tensor] = []
    labels: List[Tensor] = []

    for sample in batch:
        utt_ids.append(sample["utt_id"])
        transcripts.append(sample["transcript"])
        speakers.append(sample["speaker"])
        waveforms.append(sample["waveform"])
        lengths.append(sample["length"])
        labels.append(sample["labels"])

    max_len: int = max(w.shape[0] for w in waveforms)

    padded_waveforms = torch.stack(
        [F.pad(w, (0, max_len - w.shape[0])) for w in waveforms]
    )  # (B, T_max)

    length_tensor = torch.stack(lengths)  # (B,)

    # Build boolean attention mask: True where real, False where padded.
    attention_mask = torch.arange(max_len).unsqueeze(0) < length_tensor.unsqueeze(1)
    attention_mask = attention_mask.long()  # (B, T_max)

    return {
        "waveforms": padded_waveforms,
        "attention_mask": attention_mask,
        "lengths": length_tensor,
        "labels": torch.stack(labels),
        "utt_ids": utt_ids,
        "transcripts": transcripts,
        "speakers": speakers,
    }
