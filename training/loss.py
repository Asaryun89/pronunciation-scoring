from __future__ import annotations

import torch
from torch import nn


class PronunciationLoss(nn.Module):
    """Combine utterance-level and phoneme-level regression losses."""

    def __init__(self, phoneme_weight: float = 1.0) -> None:
        super().__init__()
        self.phoneme_weight = phoneme_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        utterance_pred: torch.Tensor,
        utterance_target: torch.Tensor,
        phoneme_pred: torch.Tensor,
        phoneme_target: torch.Tensor,
        seg_lengths: torch.Tensor,
    ) -> torch.Tensor:
        utterance_loss = self.mse(utterance_pred, utterance_target)

        if phoneme_pred.numel() == 0:
            return utterance_loss

        mask = torch.arange(phoneme_pred.size(1), device=phoneme_pred.device)[None, :] < seg_lengths[:, None]
        diff = (phoneme_pred - phoneme_target) ** 2
        masked = diff * mask.float()
        denom = mask.sum().clamp_min(1)
        phoneme_loss = masked.sum() / denom

        return utterance_loss + self.phoneme_weight * phoneme_loss
