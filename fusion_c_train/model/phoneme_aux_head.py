"""
Auxiliary phoneme prediction head for Fusion-C fine-tuning.

Applied to f₁ output at masked positions during training only.
Forces the f₁ encoder to preserve phoneme discriminability rather than
collapsing representations during fine-tuning.  Not used at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def generate_finetune_mask(
    x:           Tensor,
    mask_prob:   float = 0.015,
    mask_length: int   = 5,
) -> Tensor:
    """
    Generate random contiguous-span masks for the auxiliary head.

    Lighter than pre-training masking (mask_prob=0.015 vs 0.065,
    mask_length=5 vs 10) so it does not degrade the main scoring task.

    Args:
        x:           [B, T, H] — frame sequence (used for shape only)
        mask_prob:   Fraction of frames to mask per utterance.
        mask_length: Contiguous span length in frames.

    Returns:
        [B, T] bool — True = masked position
    """
    B, T, _ = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    if mask_prob <= 0.0 or T <= mask_length:
        return mask
    n_masks = max(1, int(mask_prob * T))
    for b in range(B):
        starts = torch.randint(0, T - mask_length, (n_masks,), device=x.device)
        for s in starts:
            mask[b, s : s + mask_length] = True
    return mask


class PhonemeAuxHead(nn.Module):
    """
    Predicts k-means cluster IDs at masked f₁ positions.

    Args:
        hidden_dim:  f₁ hidden size (e.g. 768 for HuBERT-base, 1024 for large).
        n_clusters:  k-means vocabulary size (must match kmeans.n_clusters_hi).

    Forward (training only):
        f1_output: [B, T, H]
        mask:      [B, T] bool — True = masked position to predict on
        returns:   [N_masked, n_clusters] logits
    """

    def __init__(self, hidden_dim: int = 768, n_clusters: int = 100) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_clusters),
        )

    def forward(self, f1_output: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            f1_output: [B, T, H]
            mask:      [B, T] bool

        Returns:
            [N_masked, n_clusters]
        """
        masked_features = f1_output[mask]    # [N_masked, H]
        return self.head(masked_features)
