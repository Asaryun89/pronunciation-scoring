"""
Cross-attention fusion module for pronunciation scoring.

Upgrade 2: CrossAttentionFusion now accepts a per-token text sequence
[B, N, 256] as key/value instead of a single global vector [B, 1, 256].
Audio frames in the query stream can therefore attend to individual subword
tokens rather than a collapsed global embedding.

    Query:  pre_fused[3]   [B, T', 256]  — audio (total stream)
    Key/V:  text_feats     [B, N,  256]  — token sequence from TokenTextProjection
    Mask:   key_pad_mask   [B, N]  bool  — True = padding to ignore
    Output: [B, T', 256]  — fused features with residual + LayerNorm

TextProjection is preserved as a legacy class for backward compatibility
(used by tests that test it in isolation).  It is no longer wired into
CrossAttentionFusion; that role is now played by TokenTextProjection in
model/text_projection.py.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# TextProjection — legacy single-vector projection (kept for test compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class TextProjection(nn.Module):
    """
    Projects a single global text embedding to the shared proj_dim.

    Legacy: used in tests and pre-Upgrade-2 checkpoints.  No longer called
    by CrossAttentionFusion; replaced by TokenTextProjection for token-level
    cross-attention (Upgrade 2).

    Input:  [B, in_dim]  — mean-pool text embedding
    Output: [B, 1, out_dim]
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, text_emb: Tensor) -> Tensor:
        """
        Args:
            text_emb: [B, in_dim]

        Returns:
            [B, 1, out_dim]
        """
        # TODO Upgrade 2: replace single-vector [B,1,256] with
        # per-token sequence [B,N,256] for phoneme-granular cross-attention.
        return self.norm(self.proj(text_emb)).unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# CrossAttentionFusion — Upgrade 2: per-token key/value
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Audio Q attends to per-token text K/V via multi-head cross-attention.

    Upgrade 2 change: accepts a full token sequence [B, N, proj_dim] as
    key_value (plus an optional key_padding_mask) so audio frames can
    attend to individual subword tokens.

    Architecture:
        attn_out = MHA(Q=query, K=key_value, V=key_value,
                       key_padding_mask=key_padding_mask)
        out      = LayerNorm(query + attn_out)   # residual

    Args:
        cfg: Full config dict (reads model.proj_dim, fusion_heads,
             fusion_dropout).  text_encoder_dim is no longer needed here;
             TextProjection has been moved to TokenTextProjection.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg     = cfg["model"]
        proj_dim = mcfg["proj_dim"]

        self.mha = nn.MultiheadAttention(
            embed_dim  = proj_dim,
            num_heads  = mcfg["fusion_heads"],
            dropout    = mcfg["fusion_dropout"],
            batch_first= True,   # Q/K/V all [B, seq, dim]
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(
        self,
        query:             Tensor,                    # [B, T', proj_dim]
        key_value:         Tensor,                    # [B, N,  proj_dim]
        key_padding_mask:  Optional[Tensor] = None,   # [B, N]  bool
    ) -> Tensor:
        """
        Args:
            query:            [B, T', proj_dim]  audio pre-fused features (Q)
            key_value:        [B, N,  proj_dim]  per-token text features (K/V)
            key_padding_mask: [B, N]  bool  True = padding position to ignore

        Returns:
            [B, T', proj_dim] — fused features with residual + LayerNorm
        """
        attn_out, _ = self.mha(
            query             = query,
            key               = key_value,
            value             = key_value,
            key_padding_mask  = key_padding_mask,
        )                                               # [B, T', proj_dim]
        return self.norm(query + attn_out)
