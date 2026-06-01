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

from model.fusion_head import SCORE_DIMS, FusionScoringHead

# Default dims matching finetune_config.yaml
SPEECH_DIM = 1024
TEXT_DIM   = 1024
HIDDEN     = 512
N_SCORES   = 4
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
# Existing tests (must still pass)
# ─────────────────────────────────────────────────────────────────────────────

def test_output_shape(head: FusionScoringHead) -> None:
    """Output must be [B, 4]."""
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
    gate projection and at least one per-dimension regressor.
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

    # At least one parameter across all regressors must have a non-zero gradient.
    for name in head.dim_names:
        for p in head.regressors[name].parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                return
    pytest.fail("No regressor weight has a non-zero gradient after backward.")


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
    """Verify that head was built for exactly N_SCORES = 4 outputs."""
    sr, te = _dummy_inputs()
    out    = head(sr, te)
    assert out.shape[-1] == 4, f"Expected 4 score dims, got {out.shape[-1]}"


# ─────────────────────────────────────────────────────────────────────────────
# New tests for per-dimension architecture
# ─────────────────────────────────────────────────────────────────────────────

def test_per_dim_independent_gradients(head: FusionScoringHead) -> None:
    """
    Backpropagating through a single regressor must leave all other
    regressors' parameters with grad=None.  This proves independence.
    """
    head.train()
    sr, te = _dummy_inputs()

    # Compute shared fused representation manually.
    concat = torch.cat([sr, te], dim=-1)
    fused  = torch.sigmoid(head.gate_proj(concat)) * concat

    head.zero_grad()

    # Only run the "total" regressor — others are never called.
    score = head.regressors["total"](fused)
    score.sum().backward()

    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in head.regressors["total"].parameters()
    ), "total regressor has no gradient after backward"

    for name in ["accuracy", "fluency", "prosodic"]:
        for p in head.regressors[name].parameters():
            assert p.grad is None, (
                f"'{name}' regressor unexpectedly has gradient "
                f"(independence violated)"
            )


def test_output_bounded(head: FusionScoringHead) -> None:
    """All outputs must be strictly within [0.0, 1.0] for varied inputs."""
    for _ in range(100):
        sr = torch.randn(BATCH, SPEECH_DIM) * 5.0  # large magnitude stress test
        te = torch.randn(BATCH, TEXT_DIM)   * 5.0
        out = head(sr, te)
        assert out.min().item() >= 0.0, f"output below 0: {out.min().item()}"
        assert out.max().item() <= 1.0, f"output above 1: {out.max().item()}"


def test_dim_names_order(head: FusionScoringHead) -> None:
    """dim_names must match SCORE_DIMS in fixed canonical order."""
    assert head.dim_names == SCORE_DIMS
    assert head.dim_names == ["total", "accuracy", "fluency", "prosodic"]
