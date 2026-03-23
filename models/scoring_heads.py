"""
Scoring-head building blocks used by HubertMultiTask.

Architecture (matching the diagram):
  CrossAttentionFusion  — Audio Q, Text K/V  (or both ways)
  MLPScoringHead        — (FC → ReLU → Dropout) × hidden_layers → FC → Sigmoid × num_aspects
"""

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Cross-Attention Fusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Fuse audio and text streams via cross-attention.

    Audio embeddings are the queries; text embeddings supply the keys and
    values.  A residual connection and LayerNorm are applied after attention
    so this block can be stacked or placed before a Transformer encoder.

    Args
    ----
    d_model   : common projection dimension for audio and text
    num_heads : number of attention heads
    dropout   : attention + residual dropout rate
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        audio_emb:        torch.Tensor,           # (B, T, d_model)  — queries
        text_emb:         torch.Tensor,           # (B, L, d_model)  — keys & values
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L)  True = ignore
    ) -> torch.Tensor:                            # (B, T, d_model)
        attn_out, _ = self.attn(
            audio_emb, text_emb, text_emb,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(audio_emb + self.dropout(attn_out))


# ---------------------------------------------------------------------------
# MLP Scoring Head
# ---------------------------------------------------------------------------

class MLPScoringHead(nn.Module):
    """
    Deep MLP scoring head with configurable number of hidden layers.

    Maps an utterance embedding to per-aspect scores in [0, 1].

    Pipeline: [ FC(d_model→d_model) → ReLU → Dropout ] × hidden_layers
              → FC(d_model → num_aspects) → Sigmoid

    Args
    ----
    d_model       : input (and hidden) dimension
    num_aspects   : number of output scores (default 5 for SpeechOcean)
    hidden_layers : number of hidden FC→ReLU→Dropout blocks (default 1)
    dropout       : dropout rate inside each hidden block
    """

    def __init__(
        self,
        d_model:       int   = 256,
        num_aspects:   int   = 5,
        hidden_layers: int   = 1,
        dropout:       float = 0.1,
    ):
        super().__init__()
        layers: list = []
        for _ in range(hidden_layers):
            layers += [nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(d_model, num_aspects), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d_model) → (B, num_aspects)  values in [0, 1]"""
        return self.net(x)
