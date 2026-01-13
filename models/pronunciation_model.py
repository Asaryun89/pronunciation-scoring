from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_dim: int
    ssl_dim: int
    pron_dim: int
    transformer_heads: int
    transformer_layers: int
    dropout: float


class PronunciationModel(nn.Module):
    """SSL encoder + pronunciation encoder + phoneme scoring head + aggregation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ssl_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.ssl_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.ssl_dim),
        )

        self.pron_conv = nn.Sequential(
            nn.Conv1d(config.ssl_dim, config.pron_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.pron_dim,
            nhead=config.transformer_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.pron_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
        )

        self.frame_scorer = nn.Sequential(
            nn.Linear(config.pron_dim, config.pron_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.pron_dim, 1),
        )

    def _segment_mean(self, frame_scores: torch.Tensor, phoneme_lengths: torch.Tensor) -> torch.Tensor:
        """Pool frame-level scores into phoneme-level scores using alignment lengths."""
        batch_size, max_frames = frame_scores.shape
        max_phonemes = phoneme_lengths.shape[1]
        phoneme_scores = torch.zeros(batch_size, max_phonemes, device=frame_scores.device)
        for i in range(batch_size):
            offset = 0
            for j in range(max_phonemes):
                length = int(phoneme_lengths[i, j].item())
                if length <= 0:
                    break
                end = min(offset + length, max_frames)
                if end > offset:
                    phoneme_scores[i, j] = frame_scores[i, offset:end].mean()
                offset = end
        return phoneme_scores

    def forward(
        self,
        features: torch.Tensor,
        feat_lengths: torch.Tensor,
        phoneme_lengths: torch.Tensor,
    ) -> dict:
        """
        Args:
            features: (B, T, F)
            feat_lengths: (B,)
            phoneme_lengths: (B, P)
        Returns:
            dict with frame_scores, phoneme_scores, utterance_score, and aggregates
        """
        x = self.ssl_encoder(features)

        x = x.transpose(1, 2)  # (B, C, T)
        x = self.pron_conv(x)
        x = x.transpose(1, 2)  # (B, T, C)

        mask = torch.arange(x.size(1), device=x.device)[None, :] < feat_lengths[:, None]
        x = self.pron_transformer(x, src_key_padding_mask=~mask)

        frame_scores = self.frame_scorer(x).squeeze(-1)
        frame_scores = frame_scores.masked_fill(~mask, 0.0)

        phoneme_scores = self._segment_mean(frame_scores, phoneme_lengths)
        phoneme_mask = phoneme_lengths > 0
        phoneme_sum = (phoneme_scores * phoneme_mask).sum(dim=1)
        phoneme_count = phoneme_mask.sum(dim=1).clamp(min=1)
        utterance_score = phoneme_sum / phoneme_count

        phoneme_mean = utterance_score
        phoneme_std = torch.sqrt(
            ((phoneme_scores - phoneme_mean[:, None]) ** 2 * phoneme_mask).sum(dim=1) / phoneme_count
        )
        fluency_score = (100.0 - torch.clamp(phoneme_std * 10.0, max=100.0)).clamp(min=0.0)
        prosody_score = 0.5 * (phoneme_mean + fluency_score)

        return {
            "frame_scores": frame_scores,
            "phoneme_scores": phoneme_scores,
            "phoneme_mask": phoneme_mask,
            "utterance_score": utterance_score,
            "accuracy_score": phoneme_mean,
            "fluency_score": fluency_score,
            "prosody_score": prosody_score,
        }
