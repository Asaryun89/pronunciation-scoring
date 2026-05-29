"""
Gated fusion scoring head for Fusion-C pronunciation assessment.

Combines mean-pooled speech representations (from f₃) with L2-normalised
BGE text embeddings via a learned element-wise gate, then regresses to
5 MOS-style pronunciation scores.

Score dimension order (output indices):
    0 → total
    1 → accuracy
    2 → fluency
    3 → prosodic
    4 → completeness

Standalone import:
    from model.fusion_head import FusionScoringHead
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FusionScoringHead(nn.Module):
    """
    Gated multimodal fusion → pronunciation score regression.

    Architecture
    ────────────
    1. Concatenate speech_rep [B, speech_dim] and text_emb [B, text_dim]
       → concat [B, speech_dim + text_dim]

    2. Element-wise gate:
         gate  = sigmoid( Linear(in_dim, in_dim)(concat) )
         fused = gate ⊙ concat                              [B, in_dim]

    3. MLP:
         Linear(in_dim, hidden) → LayerNorm → ReLU → Dropout
         → Linear(hidden, hidden//2) → ReLU
         → Linear(hidden//2, n_scores)
         → Sigmoid                                          [B, n_scores]

    Args:
        speech_dim: Dimensionality of the speech representation (e.g. 1024).
        text_dim:   Dimensionality of the text embedding (e.g. 384).
        hidden:     Internal MLP width (e.g. 512).
        n_scores:   Number of output score dimensions (5).
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
        in_dim = speech_dim + text_dim

        # Learned element-wise gate over the concatenated representation.
        self.gate_proj = nn.Linear(in_dim, in_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_scores),
        )
        # Sigmoid bounds predictions to (0, 1); targets are also normalised to [0, 1].
        self.out_act = nn.Sigmoid()

    def forward(self, speech_rep: Tensor, text_emb: Tensor) -> Tensor:
        """
        Args:
            speech_rep: [B, speech_dim] — mean-pooled f₃ output
            text_emb:   [B, text_dim]   — BGE [CLS] embedding (L2-normalised)

        Returns:
            scores: [B, n_scores] in (0, 1)
                    dim 0 = total, 1 = accuracy, 2 = fluency,
                    3 = prosodic, 4 = completeness
        """
        concat = torch.cat([speech_rep, text_emb], dim=-1)  # [B, in_dim]
        gate   = torch.sigmoid(self.gate_proj(concat))       # [B, in_dim]
        fused  = gate * concat                               # [B, in_dim]
        return self.out_act(self.mlp(fused))                 # [B, n_scores]
