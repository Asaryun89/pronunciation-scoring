"""
Tests for model/cross_attention_fusion.py — CrossAttentionFusion.

Updated for Upgrade 2:
  - CrossAttentionFusion now accepts per-token key/value [B, N, proj_dim]
    plus an optional key_padding_mask, instead of a raw text embedding.
  - self.cross_attn renamed to self.mha.
  - TextProjection is preserved as a legacy class and still tested in
    isolation, but is no longer called by CrossAttentionFusion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.cross_attention_fusion import CrossAttentionFusion, TextProjection

PROJ_DIM = 256
TEXT_DIM = 1024
N_HEADS  = 4
BATCH    = 4
T        = 50   # audio sequence length
N_TOK    = 12   # text token count


def _make_cfg(
    proj_dim: int = PROJ_DIM,
    text_dim: int = TEXT_DIM,
    n_heads:  int = N_HEADS,
) -> dict:
    return {
        "model": {
            "proj_dim":         proj_dim,
            "text_encoder_dim": text_dim,
            "fusion_heads":     n_heads,
            "fusion_dropout":   0.0,   # deterministic for tests
        }
    }


@pytest.fixture
def fusion() -> CrossAttentionFusion:
    return CrossAttentionFusion(_make_cfg()).eval()


# ─────────────────────────────────────────────────────────────────────────────
# TextProjection — legacy single-vector projection (still exported)
# ─────────────────────────────────────────────────────────────────────────────

def test_text_projection_output_shape() -> None:
    """TextProjection must output [B, 1, proj_dim] (legacy single-vector path)."""
    proj = TextProjection(TEXT_DIM, PROJ_DIM).eval()
    text = torch.randn(BATCH, TEXT_DIM)
    out  = proj(text)
    assert out.shape == (BATCH, 1, PROJ_DIM), f"Got {tuple(out.shape)}"


def test_text_projection_layernorm_applied() -> None:
    """Output should not have the same variance as input (LayerNorm normalises)."""
    proj = TextProjection(TEXT_DIM, PROJ_DIM).eval()
    text = torch.randn(BATCH, TEXT_DIM) * 100  # large-scale input
    out  = proj(text)
    var = out.squeeze(1).var(dim=-1).mean().item()
    assert var < 10.0, f"Variance {var:.2f} too large — LayerNorm may not be applied"


# ─────────────────────────────────────────────────────────────────────────────
# CrossAttentionFusion — Upgrade 2: per-token K/V interface
# ─────────────────────────────────────────────────────────────────────────────

def test_fusion_has_mha_attribute(fusion: CrossAttentionFusion) -> None:
    """CrossAttentionFusion must expose self.mha (renamed from cross_attn)."""
    import torch.nn as nn
    assert hasattr(fusion, "mha"), "CrossAttentionFusion must have self.mha"
    assert isinstance(fusion.mha, nn.MultiheadAttention)


def test_fusion_output_shape(fusion: CrossAttentionFusion) -> None:
    """Output must be [B, T, proj_dim] — same shape as query."""
    audio_q   = torch.randn(BATCH, T, PROJ_DIM)
    text_kv   = torch.randn(BATCH, N_TOK, PROJ_DIM)
    out       = fusion(audio_q, text_kv)
    assert out.shape == (BATCH, T, PROJ_DIM), f"Got {tuple(out.shape)}"


def test_residual_preserves_audio_shape(fusion: CrossAttentionFusion) -> None:
    """Output must have same shape as query (residual connection)."""
    audio_q = torch.randn(2, 30, PROJ_DIM)
    text_kv = torch.randn(2, N_TOK, PROJ_DIM)
    out     = fusion(audio_q, text_kv)
    assert out.shape == audio_q.shape


def test_key_padding_mask_accepted(fusion: CrossAttentionFusion) -> None:
    """CrossAttentionFusion must accept and apply key_padding_mask."""
    B = 2
    audio_q   = torch.randn(B, T, PROJ_DIM)
    text_kv   = torch.randn(B, N_TOK, PROJ_DIM)
    # Mask out the last 4 tokens
    kpm = torch.zeros(B, N_TOK, dtype=torch.bool)
    kpm[:, -4:] = True
    out_masked   = fusion(audio_q, text_kv, key_padding_mask=kpm)
    out_unmasked = fusion(audio_q, text_kv, key_padding_mask=None)
    assert out_masked.shape == (B, T, PROJ_DIM)
    # Masked and unmasked outputs should differ (masking changes attention).
    assert not torch.allclose(out_masked, out_unmasked, atol=1e-5), \
        "Masked and unmasked outputs should differ"


def test_gradient_flows_through_fusion() -> None:
    """Backward must reach mha.in_proj_weight."""
    fusion  = CrossAttentionFusion(_make_cfg()).train()
    audio_q = torch.randn(2, 20, PROJ_DIM, requires_grad=True)
    text_kv = torch.randn(2, N_TOK, PROJ_DIM)
    out     = fusion(audio_q, text_kv)
    out.sum().backward()
    assert fusion.mha.in_proj_weight.grad is not None, \
        "mha.in_proj_weight has no grad after backward"
    assert fusion.mha.in_proj_weight.grad.abs().sum() > 0


def test_batch_size_one(fusion: CrossAttentionFusion) -> None:
    audio_q = torch.randn(1, T, PROJ_DIM)
    text_kv = torch.randn(1, N_TOK, PROJ_DIM)
    out     = fusion(audio_q, text_kv)
    assert out.shape == (1, T, PROJ_DIM)


def test_different_proj_dims() -> None:
    """Fusion should work for non-default proj_dim values."""
    cfg   = _make_cfg(proj_dim=128, text_dim=512, n_heads=2)
    f     = CrossAttentionFusion(cfg).eval()
    audio = torch.randn(3, 25, 128)
    text  = torch.randn(3, 8, 128)   # token sequence in proj_dim space
    out   = f(audio, text)
    assert out.shape == (3, 25, 128)
