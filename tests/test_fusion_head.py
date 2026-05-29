"""
Unit tests for model/fusion_head.py — FusionScoringHead.

Run:
    cd multi_res_hubert
    pytest tests/test_fusion_head.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.fusion_head import FusionScoringHead

# Default dims matching finetune_config.yaml
SPEECH_DIM = 1024
TEXT_DIM   = 384
HIDDEN     = 512
N_SCORES   = 5
DROPOUT    = 0.1
BATCH      = 8


@pytest.fixture
def head() -> FusionScoringHead:
    return FusionScoringHead(
        speech_dim = SPEECH_DIM,
        text_dim   = TEXT_DIM,
        hidden     = HIDDEN,
        n_scores   = N_SCORES,
        dropout    = DROPOUT,
    ).eval()


def _dummy_inputs(
    batch: int = BATCH,
    speech_dim: int = SPEECH_DIM,
    text_dim:   int = TEXT_DIM,
) -> tuple:
    speech_rep = torch.randn(batch, speech_dim)
    text_emb   = torch.randn(batch, text_dim)
    return speech_rep, text_emb


# ─────────────────────────────────────────────────────────────────────────────

def test_output_shape(head: FusionScoringHead) -> None:
    """Output must be [B, 5]."""
    sr, te = _dummy_inputs()
    out    = head(sr, te)
    assert out.shape == (BATCH, N_SCORES), (
        f"Expected ({BATCH}, {N_SCORES}), got {tuple(out.shape)}"
    )


def test_output_range(head: FusionScoringHead) -> None:
    """Sigmoid output must be in (0, 1)."""
    sr, te = _dummy_inputs()
    out    = head(sr, te)
    assert out.min().item() >= 0.0, f"min={out.min().item():.6f} < 0"
    assert out.max().item() <= 1.0, f"max={out.max().item():.6f} > 1"


def test_gate_values_in_0_1(head: FusionScoringHead) -> None:
    """Gate activations (sigmoid outputs) must lie in [0, 1]."""
    sr, te  = _dummy_inputs()
    concat  = torch.cat([sr, te], dim=-1)
    gate    = torch.sigmoid(head.gate_proj(concat))
    assert gate.min().item() >= 0.0
    assert gate.max().item() <= 1.0


def test_gradient_flows_into_head_params(head: FusionScoringHead) -> None:
    """
    A backward pass through the head must populate gradients in both the
    gate projection and the MLP parameters.
    """
    head.train()
    sr, te = _dummy_inputs()
    sr     = sr.requires_grad_(False)
    te     = te.requires_grad_(False)

    out  = head(sr, te)
    loss = out.sum()
    loss.backward()

    gate_grad = head.gate_proj.weight.grad
    assert gate_grad is not None, "gate_proj.weight has no gradient after backward"
    assert gate_grad.abs().sum().item() > 0, "gate_proj.weight gradient is all zeros"

    # Check at least one MLP linear layer has a gradient.
    for module in head.mlp:
        if hasattr(module, "weight") and module.weight.grad is not None:
            if module.weight.grad.abs().sum().item() > 0:
                return   # found a live gradient
    pytest.fail("No MLP weight has a non-zero gradient after backward.")


def test_gradient_does_not_flow_into_detached_inputs(
    head: FusionScoringHead,
) -> None:
    """
    When speech_rep and text_emb are detached, backward should NOT
    accumulate gradients on those tensors.
    """
    head.train()
    sr = torch.randn(BATCH, SPEECH_DIM, requires_grad=True)
    te = torch.randn(BATCH, TEXT_DIM,   requires_grad=True)

    sr_detached = sr.detach()
    te_detached = te.detach()

    out  = head(sr_detached, te_detached)
    loss = out.sum()
    loss.backward()

    assert sr.grad is None, "Gradient flowed back into detached speech_rep source."
    assert te.grad is None, "Gradient flowed back into detached text_emb source."


def test_batch_size_one(head: FusionScoringHead) -> None:
    """Model must handle batch size of 1 without errors."""
    sr, te = _dummy_inputs(batch=1)
    out    = head(sr, te)
    assert out.shape == (1, N_SCORES)


def test_score_dim_count(head: FusionScoringHead) -> None:
    """Verify that head was built for exactly N_SCORES = 5 outputs."""
    sr, te = _dummy_inputs()
    out    = head(sr, te)
    assert out.shape[-1] == 5, f"Expected 5 score dims, got {out.shape[-1]}"
