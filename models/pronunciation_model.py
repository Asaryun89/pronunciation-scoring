from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class ModelConfig:
    input_dim: int
    cnn_channels: list
    lstm_hidden: int
    lstm_layers: int
    dropout: float


class PronunciationModel(nn.Module):
    """CNN + BiLSTM pronunciation scoring model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers = []
        in_ch = config.input_dim
        for ch in config.cnn_channels:
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_ch = ch
        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )

        lstm_out = config.lstm_hidden * 2
        self.segment_head = nn.Sequential(
            nn.Linear(lstm_out, lstm_out),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_out, 1),
        )
        self.utterance_head = nn.Sequential(
            nn.Linear(lstm_out, lstm_out),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_out, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        feat_lengths: torch.Tensor,
        phoneme_segments: torch.Tensor,
        seg_lengths: torch.Tensor,
    ) -> dict:
        """
        Args:
            features: (B, T, F)
            feat_lengths: (B,)
            phoneme_segments: (B, S, 2) frame indices [start, end)
            seg_lengths: (B,)
        Returns:
            dict with utterance_score and phoneme_scores
        """
        x = features.transpose(1, 2)  # (B, F, T)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (B, T, C)

        packed = pack_padded_sequence(x, feat_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Utterance score from masked mean pooling
        mask = torch.arange(lstm_out.size(1), device=lstm_out.device)[None, :] < feat_lengths[:, None]
        masked = lstm_out * mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / feat_lengths.unsqueeze(1).clamp_min(1)
        utterance_score = self.utterance_head(pooled).squeeze(1)

        # Phoneme-level scores via segment pooling
        batch_scores = []
        for b in range(lstm_out.size(0)):
            segs = phoneme_segments[b, : seg_lengths[b]]
            scores = []
            for s, e in segs.tolist():
                s = max(0, min(s, int(feat_lengths[b].item())))
                e = max(s + 1, min(e, int(feat_lengths[b].item())))
                seg_vec = lstm_out[b, s:e].mean(dim=0)
                score = self.segment_head(seg_vec).squeeze(0)
                scores.append(score)
            if scores:
                batch_scores.append(torch.stack(scores))
            else:
                batch_scores.append(torch.zeros((0,), device=lstm_out.device))

        max_len = max((s.shape[0] for s in batch_scores), default=0)
        phoneme_scores = torch.zeros((lstm_out.size(0), max_len), device=lstm_out.device)
        for i, s in enumerate(batch_scores):
            if s.numel() > 0:
                phoneme_scores[i, : s.shape[0]] = s

        return {
            "utterance_score": utterance_score,
            "phoneme_scores": phoneme_scores,
        }
