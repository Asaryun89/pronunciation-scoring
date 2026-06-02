"""
MLP scoring head for pronunciation assessment.

5 independent per-dimension regressors, each with sigmoid output in (0, 1).
Multiply by 10 to get MOS scores in (0, 10).

Score dimension order (matches dataset label order):
    0 → total
    1 → accuracy
    2 → fluency
    3 → prosodic
    (completeness excluded — skewed distribution, persistent low PCC)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

SCORE_DIMS: List[str] = ["total", "accuracy", "fluency", "prosodic"]
# "completeness" excluded — skewed distribution causes persistent low PCC


class _DimHead(nn.Module):
    """Single-dimension MLP regressor with sigmoid output in (0, 1)."""

    def __init__(self, in_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """[B, in_dim] → [B, 1] in (0, 1)."""
        return torch.sigmoid(self.net(x))


class MLPScoringHead(nn.Module):
    """
    5 independent MLP regressors producing scores in [0, 1].

    Multiply the output by 10 to obtain MOS-range scores in [0, 10].

    Args:
        cfg: Full config dict (reads model.proj_dim and model.scoring_dropout).
    """

    SCORE_DIMS: List[str] = SCORE_DIMS

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg    = cfg["model"]
        in_dim  = mcfg["proj_dim"]
        dropout = mcfg.get("scoring_dropout", 0.1)
        self.heads = nn.ModuleList(
            [_DimHead(in_dim, dropout) for _ in SCORE_DIMS]
        )

    @property
    def n_scores(self) -> int:
        return len(SCORE_DIMS)

    def forward(self, pooled: Tensor) -> Tensor:
        """
        Args:
            pooled: [B, proj_dim] — mean-pooled post-fusion features

        Returns:
            [B, 5] — per-dimension scores in (0, 1);
                     multiply by 10 for MOS display.
        """
        return torch.cat([h(pooled) for h in self.heads], dim=-1)
