from __future__ import annotations

import torch
from torch import nn


class PronunciationLoss(nn.Module):
    """Multi-level regression loss."""

    def __init__(self, utterance_weight: float = 1.0, phoneme_weight: float = 1.0) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.utterance_weight = utterance_weight
        self.phoneme_weight = phoneme_weight

    def forward(
        self,
        utterance_pred: torch.Tensor,
        utterance_target: torch.Tensor,
        phoneme_pred: torch.Tensor,
        phoneme_target: torch.Tensor,
        phoneme_mask: torch.Tensor,
    ) -> torch.Tensor:
        utterance_loss = self.mse(utterance_pred, utterance_target)
        diff = (phoneme_pred - phoneme_target) ** 2
        mask = phoneme_mask.to(diff.dtype)
        diff = diff * mask
        denom = mask.sum().clamp(min=1)
        phoneme_loss = diff.sum() / denom
        return self.utterance_weight * utterance_loss + self.phoneme_weight * phoneme_loss
