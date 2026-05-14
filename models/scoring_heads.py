from __future__ import annotations

"""
CrossAttentionFusion and MLPScoringHead building blocks for pronunciation scoring.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Fuse audio embeddings with context embeddings via multi-head cross-attention.

    Query = audio frames, Key/Value = context embeddings.  The fused output is
    formed via residual connection and LayerNorm.

    Args:
        d_model: Embedding dimension shared by audio and context branches.
        num_heads: Number of attention heads.
        dropout: Dropout applied to the attention output before the residual add.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        audio_emb: torch.Tensor,
        context_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-attention fusion.

        Args:
            audio_emb: ``(B, T, D)`` audio frame embeddings used as queries.
            context_emb: ``(B, L, D)`` context embeddings used as keys and values.
            key_padding_mask: ``(B, L)`` bool mask; ``True`` = ignore that key
                position (passed directly to :class:`~torch.nn.MultiheadAttention`).

        Returns:
            ``(B, T, D)`` audio embeddings enriched with context.
        """
        attn_out, _ = self.attn(
            query=audio_emb,
            key=context_emb,
            value=context_emb,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(audio_emb + self.dropout(attn_out))


class MLPScoringHead(nn.Module):
    """MLP that maps a pooled embedding to pronunciation aspect scores.

    Architecture::

        [Linear(d_model → d_model) → ReLU → Dropout] × hidden_layers
        → Linear(d_model → num_aspects) → Sigmoid

    Args:
        d_model: Input embedding dimension.
        num_aspects: Number of output pronunciation score dimensions.
        hidden_layers: Number of hidden ``Linear → ReLU → Dropout`` blocks.
        dropout: Dropout probability inside each hidden block.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_aspects: int = 5,
        hidden_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(hidden_layers):
            layers += [nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(d_model, num_aspects), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a pooled utterance representation.

        Args:
            x: ``(B, d_model)`` pooled embedding.

        Returns:
            ``(B, num_aspects)`` scores in ``[0, 1]``.
        """
        return self.net(x)
