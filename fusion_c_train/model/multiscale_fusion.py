"""
Multi-scale feature fusion for Multi-resolution HuBERT.

Combines f₁ (fine-grained phoneme), f₂ (upsampled low-res), and f₃
(high-res final) pooled representations with learned per-scale weights.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiScaleFusion(nn.Module):
    """
    Weighted combination of three scale-specific pooled representations.

    Helps accuracy PCC by preserving fine-grained phoneme detail from f₁
    that can be diluted after the DOWN/UP bottleneck.

    Args:
        hidden_dim: Dimensionality of all three input tensors (e.g. 768).

    Forward:
        f1_pooled:           [B, H]
        f2_upsampled_pooled: [B, H]
        f3_pooled:           [B, H]
        returns:             [B, H]
    """

    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(3) / 3.0)
        self.proj          = nn.Linear(hidden_dim, hidden_dim)
        self.norm          = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        f1_pooled:           Tensor,
        f2_upsampled_pooled: Tensor,
        f3_pooled:           Tensor,
    ) -> Tensor:
        w     = F.softmax(self.scale_weights, dim=0)
        fused = w[0] * f1_pooled + w[1] * f2_upsampled_pooled + w[2] * f3_pooled
        return self.norm(self.proj(fused))

    @property
    def scale_contributions(self) -> Dict[str, float]:
        """Softmax-normalised scale weights for logging."""
        w = F.softmax(self.scale_weights.detach(), dim=0)
        return {"f1": w[0].item(), "f2": w[1].item(), "f3": w[2].item()}
