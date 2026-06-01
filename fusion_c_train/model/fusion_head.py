"""
Gated fusion scoring head for Fusion-C pronunciation assessment.

Combines mean-pooled speech representations (from f₃) with L2-normalised
Qwen3 text embeddings via a learned element-wise gate, then passes the fused
representation through 4 independent per-dimension MLP regressors.

Score dimension order (output indices):
    0 → total
    1 → accuracy
    2 → fluency
    3 → prosodic

Standalone import:
    from model.fusion_head import FusionScoringHead
"""

from __future__ import annotations

from typing import Iterator, List

import torch
import torch.nn as nn
from torch import Tensor

SCORE_DIMS: List[str] = ["total", "accuracy", "fluency", "prosodic"]
# completeness removed — skewed distribution causes persistent low PCC


class PerDimRegressor(nn.Module):
    """
    Single-dimension MLP regressor.

    Architecture:
        fused [B, concat_dim]
        → Linear(concat_dim, hidden) → LayerNorm → ReLU → Dropout
        → Linear(hidden, hidden // 2) → ReLU
        → Linear(hidden // 2, 1)
        → sigmoid → squeeze(-1) → [B]   (output in (0, 1))

    Args:
        concat_dim: Input width (speech_dim + text_dim).
        hidden:     Hidden layer width.
        dropout:    Dropout probability.
        dim_name:   Human-readable label for this dimension.
    """

    def __init__(
        self,
        concat_dim: int,
        hidden:     int,
        dropout:    float,
        dim_name:   str,
    ) -> None:
        super().__init__()
        self.dim_name = dim_name
        self.net = nn.Sequential(
            nn.Linear(concat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, fused: Tensor) -> Tensor:
        """
        Args:
            fused: [B, concat_dim]

        Returns:
            [B] — score in (0, 1)
        """
        return torch.sigmoid(self.net(fused)).squeeze(-1)


class FusionScoringHead(nn.Module):
    """
    Gated multimodal fusion → 5 independent per-dimension MLP regressors.

    Architecture
    ────────────
    1. Concatenate speech_rep [B, speech_dim] and text_emb [B, text_dim]
       → concat [B, in_dim]

    2. Shared element-wise gate (unchanged from original):
         gate  = sigmoid( Linear(in_dim, in_dim)(concat) )
         fused = gate ⊙ concat                              [B, in_dim]

    3. Four independent PerDimRegressors, one per score dimension.
       Each regressor receives the same fused tensor and produces [B].

    4. Stack outputs in SCORE_DIMS order → [B, 4]

    Args:
        speech_dim: Dimensionality of the speech representation (e.g. 1024).
        text_dim:   Dimensionality of the text embedding (e.g. 1024).
        hidden:     Internal MLP width (e.g. 512).
        n_scores:   Number of output score dimensions (must equal len(SCORE_DIMS) = 4).
        dropout:    Dropout probability in the MLP.
    """

    def __init__(
        self,
        speech_dim: int,
        text_dim:   int,
        hidden:     int,
        n_scores:   int,
        dropout:    float,
    ) -> None:
        super().__init__()
        assert n_scores == len(SCORE_DIMS), (
            f"n_scores must be {len(SCORE_DIMS)}, got {n_scores}"
        )
        in_dim = speech_dim + text_dim

        # Shared gated fusion — identical to original architecture.
        self.gate_proj = nn.Linear(in_dim, in_dim)

        # One independent regressor per score dimension.
        self.regressors: nn.ModuleDict = nn.ModuleDict({
            name: PerDimRegressor(in_dim, hidden, dropout, name)
            for name in SCORE_DIMS
        })

    @property
    def dim_names(self) -> List[str]:
        """Fixed-order list of score dimension names."""
        return list(SCORE_DIMS)

    def get_dim_params(self, dim_name: str) -> Iterator[nn.Parameter]:
        """Return parameters for a single dimension's regressor only."""
        return self.regressors[dim_name].parameters()

    def forward(self, speech_rep: Tensor, text_emb: Tensor) -> Tensor:
        """
        Args:
            speech_rep: [B, speech_dim] — mean-pooled f₃ output
            text_emb:   [B, text_dim]   — Qwen3 last-token embedding (L2-normalised)

        Returns:
            scores: [B, 4] in (0, 1)
                    dim 0 = total, 1 = accuracy, 2 = fluency, 3 = prosodic
        """
        concat = torch.cat([speech_rep, text_emb], dim=-1)  # [B, in_dim]
        gate   = torch.sigmoid(self.gate_proj(concat))       # [B, in_dim]
        fused  = gate * concat                               # [B, in_dim]

        return torch.stack(
            [self.regressors[name](fused) for name in SCORE_DIMS], dim=-1
        )  # [B, 4]
