"""
MLP scoring head for pronunciation assessment.

4 independent per-dimension regressors, each with sigmoid output in (0, 1).
Scores are in [0, 1] to match the normalised labels in SpeechOceanASRDataset
(raw scores divided by SCORE_MAX=10 before training).

Score dimension order (matches SCORE_DIMS / dataset label order):
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
    4 independent MLP regressors producing scores in [0, 1].

    Each dimension receives its own pooled feature vector from
    PronunciationScorer (dimension-conditioned audio features).

    Args:
        cfg: Full config dict (reads model.proj_dim and model.scoring_dropout).
    """

    SCORE_DIMS: List[str] = SCORE_DIMS

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg    = cfg["model"]
        in_dim  = mcfg["proj_dim"]
        dropout = mcfg.get("scoring_dropout", 0.1)
        # Named per-dimension heads — order matches SCORE_DIMS
        self.total_mlp    = _DimHead(in_dim, dropout)
        self.accuracy_mlp = _DimHead(in_dim, dropout)
        self.fluency_mlp  = _DimHead(in_dim, dropout)
        self.prosodic_mlp = _DimHead(in_dim, dropout)

    @property
    def n_scores(self) -> int:
        return len(SCORE_DIMS)

    def forward(self, pooled: Tensor) -> Tensor:
        """
        Shared-input forward for backward-compatibility (e.g. evaluation scripts
        that pass a single pooled vector).  PronunciationScorer.forward calls
        the sub-heads directly with per-dimension pooled vectors.

        Args:
            pooled: [B, proj_dim] — mean-pooled post-fusion features

        Returns:
            [B, 4] — per-dimension scores in (0, 1);
                     order: [total, accuracy, fluency, prosodic]
        """
        return torch.cat([
            self.total_mlp(pooled),
            self.accuracy_mlp(pooled),
            self.fluency_mlp(pooled),
            self.prosodic_mlp(pooled),
        ], dim=-1)
