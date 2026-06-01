"""
Per-dimension attention pooling over frame sequences.

Replaces mean-pooling with learned queries, one per scoring dimension,
so each dimension can attend to the most relevant acoustic regions.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class AttentionPooling(nn.Module):
    """
    Learnable per-dimension attention pool over f₃ frame sequence.

    Each of the n_dims scoring dimensions has its own query vector so that
    e.g. "accuracy" can focus on consonant-heavy frames while "prosodic"
    focuses on phrase-boundary and pitch-rich regions.

    Args:
        hidden_dim: Frame embedding dimensionality (e.g. 768 / 1024).
        n_dims:     Number of scoring dimensions (must equal len(SCORE_DIMS)).

    Forward:
        f3_output:       [B, T, H]  — sequence output of the f₃ encoder
        attention_mask:  [B, T] bool — True = valid frame (False = padding)
        returns:         [B, n_dims, H]
    """

    def __init__(self, hidden_dim: int = 768, n_dims: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_dims     = n_dims

        self.queries  = nn.Parameter(torch.empty(n_dims, hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        nn.init.xavier_uniform_(self.queries.unsqueeze(0))

    def forward(
        self,
        f3_output:      Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            f3_output:       [B, T, H]
            attention_mask:  [B, T] bool, True = valid

        Returns:
            [B, n_dims, H]
        """
        keys   = self.key_proj(f3_output)                          # [B, T, H]
        scores = torch.einsum("dh,bth->bdt", self.queries, keys)   # [B, D, T]
        scores = scores / (self.hidden_dim ** 0.5)

        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask[:, None, :], float("-inf"))

        weights = torch.softmax(scores, dim=-1)                    # [B, D, T]
        pooled  = torch.einsum("bdt,bth->bdh", weights, f3_output) # [B, D, H]
        return pooled
