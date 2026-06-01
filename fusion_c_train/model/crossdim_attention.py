"""
Cross-dimension attention for Fusion-C pronunciation scoring.

Lets each dimension's hidden representation attend to all other dimensions
before final score prediction, capturing inter-dimension correlations
(e.g. total score should track accuracy + fluency + prosodic).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class CrossDimAttention(nn.Module):
    """
    Transformer-style attention across scoring dimensions.

    Args:
        hidden_dim: Hidden dimensionality per dimension (e.g. 256).
        n_dims:     Number of scoring dimensions (e.g. 4).
        n_heads:    Number of attention heads (hidden_dim must be divisible).
        dropout:    Dropout rate in the attention layer.

    Forward:
        dim_features: List of n_dims tensors, each [B, hidden_dim]
        returns:      List of n_dims tensors, each [B, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_dims:     int = 4,
        n_heads:    int = 4,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.n_dims = n_dims
        self.attn   = nn.MultiheadAttention(
            hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.norm   = nn.LayerNorm(hidden_dim)
        self.ff     = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2  = nn.LayerNorm(hidden_dim)

    def forward(self, dim_features: List[Tensor]) -> List[Tensor]:
        """
        Args:
            dim_features: List[n_dims × [B, H]]

        Returns:
            List[n_dims × [B, H]]
        """
        x = torch.stack(dim_features, dim=1)      # [B, n_dims, H]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(attn_out + x)               # residual + layernorm
        x = self.norm2(self.ff(x) + x)            # FF + residual + layernorm
        return [x[:, i, :] for i in range(self.n_dims)]
