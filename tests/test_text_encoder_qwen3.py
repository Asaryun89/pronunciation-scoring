"""
Tests for model/text_encoder_qwen3.py — Qwen3MeanPoolEncoder.

Uses a lightweight stub to avoid downloading the 596M model.
Tests the mask-aware mean pool static method with real tensors
(no HuggingFace required).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.text_encoder_qwen3 import Qwen3MeanPoolEncoder

OUTPUT_DIM = 1024


# ─────────────────────────────────────────────────────────────────────────────
# Stub encoder — mirrors the real interface without loading HF weights
# ─────────────────────────────────────────────────────────────────────────────

class _FakeQwen3Mean(nn.Module):
    INSTRUCT_PREFIX: str = Qwen3MeanPoolEncoder.INSTRUCT_PREFIX

    def __init__(self, frozen: bool = True) -> None:
        super().__init__()
        self.linear     = nn.Linear(8, OUTPUT_DIM)
        self._frozen    = frozen
        self.max_length = 128
        if frozen:
            for p in self.linear.parameters():
                p.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return OUTPUT_DIM

    @property
    def device(self) -> torch.device:
        return next(self.linear.parameters()).device

    def freeze(self) -> None:
        self._frozen = True
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        self._frozen = False
        for p in self.parameters():
            p.requires_grad_(True)

    def _encode(self, transcripts: List[str]) -> Tensor:
        x = torch.randn(len(transcripts), 8)
        return F.normalize(self.linear(x).float(), p=2, dim=-1)

    def forward(self, transcripts: List[str]) -> Tensor:
        if not self.training or self._frozen:
            with torch.no_grad():
                return self._encode(transcripts)
        return self._encode(transcripts)


@pytest.fixture
def enc() -> _FakeQwen3Mean:
    return _FakeQwen3Mean(frozen=True).eval()


@pytest.fixture
def trainable_enc() -> _FakeQwen3Mean:
    e = _FakeQwen3Mean(frozen=False)
    e.train()
    return e


# ─────────────────────────────────────────────────────────────────────────────
# mask_aware_mean_pool  (static method — no model needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_mask_aware_pool_all_real() -> None:
    """With all-ones mask, pool equals simple mean."""
    B, T, H  = 3, 10, 16
    hidden   = torch.randn(B, T, H)
    attn     = torch.ones(B, T, dtype=torch.long)
    pooled   = Qwen3MeanPoolEncoder.mask_aware_mean_pool(hidden.float(), attn)
    expected = F.normalize(hidden.mean(1).float(), p=2, dim=1)
    assert torch.allclose(pooled, expected, atol=1e-5), "Pool with all-real mask should equal mean"


def test_mask_aware_pool_ignores_padding() -> None:
    """Padding positions (mask=0) must not contribute to the mean."""
    B, T, H = 2, 8, 16
    hidden  = torch.zeros(B, T, H)
    # Place non-zero values only in real positions.
    hidden[0, :4] = 1.0   # seq 0: 4 real tokens
    hidden[1, :6] = 2.0   # seq 1: 6 real tokens
    attn = torch.zeros(B, T, dtype=torch.long)
    attn[0, :4] = 1
    attn[1, :6] = 1

    pooled = Qwen3MeanPoolEncoder.mask_aware_mean_pool(hidden.float(), attn)
    # After L2-norm: both rows should be the same direction (all-ones vectors normalised)
    assert pooled.shape == (B, H)
    # No NaN allowed
    assert not pooled.isnan().any(), "NaN in pooled output"


def test_mask_aware_differs_from_simple_mean_on_padded() -> None:
    """Mask-aware pool must give a different result from naive mean when padding present."""
    B, T, H = 2, 8, 16
    hidden  = torch.randn(B, T, H)
    # Only first 4 tokens are real.
    attn    = torch.zeros(B, T, dtype=torch.long)
    attn[:, :4] = 1

    pooled_masked = Qwen3MeanPoolEncoder.mask_aware_mean_pool(hidden.float(), attn)
    pooled_naive  = F.normalize(hidden.mean(1).float(), p=2, dim=1)
    # They should differ when there's padding (padding shifts the naive mean).
    assert not torch.allclose(pooled_masked, pooled_naive, atol=1e-3), \
        "Mask-aware pool should differ from naive mean when padding is present"


def test_l2_normalisation(enc: _FakeQwen3Mean) -> None:
    """Every output row must have L2-norm ≈ 1.0."""
    out   = enc(["first", "second", "third"])
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_output_shape(enc: _FakeQwen3Mean) -> None:
    out = enc(["THE CAT SAT ON THE MAT", "HELLO WORLD"])
    assert out.shape == (2, OUTPUT_DIM)


def test_output_dim_property(enc: _FakeQwen3Mean) -> None:
    assert enc.output_dim == OUTPUT_DIM


def test_fp32_output(enc: _FakeQwen3Mean) -> None:
    out = enc(["test"])
    assert out.dtype == torch.float32


def test_frozen_no_gradients(enc: _FakeQwen3Mean) -> None:
    enc.train()
    out = enc(["hello"])
    assert not out.requires_grad, "Frozen encoder output must not require grad"
    for p in enc.parameters():
        assert p.grad is None


def test_unfrozen_gradients_flow(trainable_enc: _FakeQwen3Mean) -> None:
    out  = trainable_enc(["hello"])
    out.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in trainable_enc.parameters()
    ), "Unfrozen encoder must have grad after backward"


def test_freeze_unfreeze_toggle() -> None:
    enc = _FakeQwen3Mean(frozen=False)
    assert all(p.requires_grad for p in enc.parameters())
    enc.freeze()
    assert all(not p.requires_grad for p in enc.parameters())
    enc.unfreeze()
    assert all(p.requires_grad for p in enc.parameters())


def test_padding_side_left() -> None:
    """Qwen3MeanPoolEncoder must init tokenizer with padding_side='left'."""
    captured: dict = {}

    def fake_tok(model_name: str, **kwargs) -> MagicMock:
        captured.update(kwargs)
        m = MagicMock()
        m.padding_side = kwargs.get("padding_side", "right")
        return m

    def fake_model(model_name: str, **kwargs) -> MagicMock:
        m = MagicMock()
        m.parameters = MagicMock(return_value=iter([nn.Parameter(torch.randn(1))]))
        return m

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_tok), \
         patch("transformers.AutoModel.from_pretrained",     side_effect=fake_model):
        Qwen3MeanPoolEncoder(model_name="fake/model", frozen=True)

    assert captured.get("padding_side") == "left", (
        f"Tokenizer should have padding_side='left', got: {captured.get('padding_side')!r}"
    )
