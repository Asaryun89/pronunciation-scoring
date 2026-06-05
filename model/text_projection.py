"""
model/text_projection.py — Upgrade 2: per-token text projection.

TokenTextProjection wraps the frozen Qwen3 language model and projects its
per-token hidden states token-wise to the shared proj_dim, returning both
the token sequence and a key_padding_mask for use in CrossAttentionFusion.

Before Upgrade 2, the text path collapsed token states to a single global
mean-pool vector [B, 1024] which was then projected to [B, 1, 256] inside
CrossAttentionFusion.  Now each audio frame in the Q stream can attend to a
specific subword token rather than one collapsed representation.

Weight migration: the Linear(1024→256) and LayerNorm weights are the same
shape as the old CrossAttentionFusion.TextProjection.proj/norm, so they load
correctly from checkpoints via migrate_text_proj() when present.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Reuse the instruct prefix that Qwen3MeanPoolEncoder established.
from .text_encoder_qwen3 import Qwen3MeanPoolEncoder


class TokenTextProjection(nn.Module):
    """
    Per-token projection of Qwen3 hidden states to proj_dim.

    Instantiates a Qwen3MeanPoolEncoder internally (frozen), discards the
    mean-pool and L2-norm steps, and applies a token-wise Linear + LayerNorm
    instead.

    Args:
        cfg: Full config dict (reads model.* keys).

    Forward:
        transcripts: List[str]
        Returns:
            projected:        [B, N, proj_dim]  — token sequence
            key_padding_mask: [B, N]  bool       — True = padding position to
                                                    mask out in nn.MultiheadAttention
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg = cfg["model"]

        # ── Frozen Qwen3 language model ────────────────────────────────────
        _enc = Qwen3MeanPoolEncoder(
            model_name   = mcfg["text_encoder_name"],
            max_length   = mcfg.get("text_max_length", 128),
            frozen       = mcfg.get("freeze_text_encoder", True),
            padding_side = mcfg.get("text_padding_side", "left"),
        )
        # Register the HF AutoModel as a submodule (frozen params → no grad).
        self.text_model = _enc.model
        # The tokenizer is a plain Python object — not an nn.Module.
        self._tokenizer = _enc.tokenizer
        self.max_length = mcfg.get("text_max_length", 128)

        # ── Trainable token-wise projection ───────────────────────────────
        text_dim = mcfg["text_encoder_dim"]   # 1024 for Qwen3-0.6B
        proj_dim = mcfg["proj_dim"]           # 256 (shared dim)
        self.proj = nn.Linear(text_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return next(self.text_model.parameters()).device

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self, transcripts: List[str]
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            transcripts: List[str] of length B — raw utterance transcripts.
                         The instruct prefix is prepended automatically.

        Returns:
            projected:        [B, N, proj_dim]  float32
            key_padding_mask: [B, N]  bool  (True = padding, ignore in MHA)
        """
        texts = [Qwen3MeanPoolEncoder.INSTRUCT_PREFIX + t for t in transcripts]
        enc = self._tokenizer(
            texts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = self.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.text_model(**enc, output_hidden_states=False)

        hidden = out.last_hidden_state.float()          # [B, N, text_dim]
        projected = self.norm(self.proj(hidden))         # [B, N, proj_dim]

        # nn.MultiheadAttention convention: True = IGNORE this position.
        # tokenizer attention_mask: 1 = real, 0 = pad → invert.
        key_padding_mask = (enc["attention_mask"] == 0)  # [B, N] bool

        return projected, key_padding_mask
