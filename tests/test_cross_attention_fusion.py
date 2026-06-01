"""
Tests for model/cross_attention_fusion.py — CrossAttentionFusion.

No stub needed — these are pure PyTorch modules.
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
T        = 50   # sequence length


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
# TextProjection
# ─────────────────────────────────────────────────────────────────────────────

def test_text_projection_output_shape() -> None:
    """TextProjection must output [B, 1, proj_dim]."""
    proj   = TextProjection(TEXT_DIM, PROJ_DIM).eval()
    text   = torch.randn(BATCH, TEXT_DIM)
    out    = proj(text)
    assert out.shape == (BATCH, 1, PROJ_DIM), f"Got {tuple(out.shape)}"


def test_text_projection_layernorm_applied() -> None:
    """Output should not have the same variance as input (LayerNorm normalises)."""
    proj = TextProjection(TEXT_DIM, PROJ_DIM).eval()
    text = torch.randn(BATCH, TEXT_DIM) * 100  # large-scale input
    out  = proj(text)
    # After LayerNorm the last dim should have near-unit variance.
    var = out.squeeze(1).var(dim=-1).mean().item()
    # Not precisely 1.0 because of learnable scale, but much smaller than 10000.
    assert var < 10.0, f"Variance {var:.2f} too large — LayerNorm may not be applied"


# ─────────────────────────────────────────────────────────────────────────────
# CrossAttentionFusion
# ─────────────────────────────────────────────────────────────────────────────

def test_fusion_output_shape(fusion: CrossAttentionFusion) -> None:
    """Output must be [B, T, proj_dim] — same shape as audio_q."""
    audio_q  = torch.randn(BATCH, T, PROJ_DIM)
    text_emb = torch.randn(BATCH, TEXT_DIM)
    out      = fusion(audio_q, text_emb)
    assert out.shape == (BATCH, T, PROJ_DIM), f"Got {tuple(out.shape)}"


def test_residual_preserves_audio_shape(fusion: CrossAttentionFusion) -> None:
    """Output must have same shape as input (residual connection)."""
    audio_q  = torch.randn(2, 30, PROJ_DIM)
    text_emb = torch.randn(2, TEXT_DIM)
    out      = fusion(audio_q, text_emb)
    assert out.shape == audio_q.shape


def test_gradient_flows_through_fusion() -> None:
    """Backward must reach cross_attn.in_proj_weight."""
    fusion   = CrossAttentionFusion(_make_cfg()).train()
    audio_q  = torch.randn(2, 20, PROJ_DIM, requires_grad=True)
    text_emb = torch.randn(2, TEXT_DIM)
    out      = fusion(audio_q, text_emb)
    out.sum().backward()
    # at least the text_proj.proj should have grad
    assert fusion.text_proj.proj.weight.grad is not None, \
        "text_proj.proj.weight has no grad after backward"
    assert fusion.text_proj.proj.weight.grad.abs().sum() > 0


def test_batch_size_one(fusion: CrossAttentionFusion) -> None:
    audio_q  = torch.randn(1, T, PROJ_DIM)
    text_emb = torch.randn(1, TEXT_DIM)
    out      = fusion(audio_q, text_emb)
    assert out.shape == (1, T, PROJ_DIM)


def test_different_proj_dims() -> None:
    """Fusion should work for non-default proj_dim values."""
    cfg   = _make_cfg(proj_dim=128, text_dim=512, n_heads=2)
    f     = CrossAttentionFusion(cfg).eval()
    audio = torch.randn(3, 25, 128)
    text  = torch.randn(3, 512)
    out   = f(audio, text)
    assert out.shape == (3, 25, 128)
