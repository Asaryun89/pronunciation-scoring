"""
Cross-attention fusion module for pronunciation scoring.

Audio frames (Q) attend to a single text token (K/V):
    AudioEncoder output [B, T, 256] × Qwen3 embedding [B, 1024]
    → CrossAttentionFusion → [B, T, 256]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TextProjection(nn.Module):
    """
    Projects Qwen3 text embedding to the shared proj_dim for K/V.

    Linear(text_dim, proj_dim) + LayerNorm, then unsqueeze to [B, 1, proj_dim]
    so the text acts as a single Key/Value token in cross-attention.
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
        return self.norm(self.proj(text_emb)).unsqueeze(1)


class CrossAttentionFusion(nn.Module):
    """
    Audio Q attends to text K/V via multi-head cross-attention.

    Architecture:
        text_kv  = TextProjection(text_emb)        # [B, 1, proj_dim]
        attn_out = MHA(Q=audio_q, K=text_kv, V=text_kv)
        out      = LayerNorm(audio_q + attn_out)   # residual

    Args:
        cfg: Full config dict (reads model.text_encoder_dim, proj_dim,
             fusion_heads, fusion_dropout).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg     = cfg["model"]
        proj_dim = mcfg["proj_dim"]
        text_dim = mcfg["text_encoder_dim"]

        self.text_proj = TextProjection(text_dim, proj_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim  = proj_dim,
            num_heads  = mcfg["fusion_heads"],
            dropout    = mcfg["fusion_dropout"],
            batch_first= True,
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, audio_q: Tensor, text_emb: Tensor) -> Tensor:
        """
        Args:
            audio_q:  [B, T, proj_dim] — pre-fusion audio features (Query)
            text_emb: [B, text_dim]    — Qwen3 mean-pool embedding

        Returns:
            [B, T, proj_dim] — fused features with residual + norm
        """
        text_kv         = self.text_proj(text_emb)           # [B, 1, proj_dim]
        attn_out, _     = self.cross_attn(
            query = audio_q,
            key   = text_kv,
            value = text_kv,
        )                                                     # [B, T, proj_dim]
        return self.norm(audio_q + attn_out)
