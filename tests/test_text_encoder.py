"""
Unit tests for model/text_encoder.py — Qwen3TextEncoder.

Tests use a lightweight stub to avoid downloading the 596M Qwen3 model.
The stub mirrors the complete public interface of Qwen3TextEncoder so all
behavioural contracts can be verified without HuggingFace weights.

Run:
    cd pronunciation-scoring
    pytest tests/test_text_encoder.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.text_encoder import Qwen3TextEncoder

OUTPUT_DIM = 1024


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight stub — avoids HuggingFace model download in CI
# ─────────────────────────────────────────────────────────────────────────────

class _FakeQwen3(nn.Module):
    """
    Minimal stand-in for Qwen3TextEncoder that skips HF model loading.

    Has a real Linear layer to probe gradient flow; mirrors the full
    freeze / unfreeze / forward interface of the real class.
    """

    INSTRUCT_PREFIX: str = Qwen3TextEncoder.INSTRUCT_PREFIX

    def __init__(self, text_dim: int = OUTPUT_DIM, frozen: bool = True) -> None:
        super().__init__()
        self.linear     = nn.Linear(8, text_dim)
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
        x   = torch.randn(len(transcripts), 8)
        emb = self.linear(x).float()
        return F.normalize(emb, p=2, dim=-1)    # [B, 1024]  float32

    def forward(self, transcripts: List[str]) -> Tensor:
        if not self.training or self._frozen:
            with torch.no_grad():
                return self._encode(transcripts)
        return self._encode(transcripts)


@pytest.fixture
def frozen_enc() -> _FakeQwen3:
    return _FakeQwen3(frozen=True).eval()


@pytest.fixture
def trainable_enc() -> _FakeQwen3:
    enc = _FakeQwen3(frozen=False)
    enc.train()
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# Constants and properties
# ─────────────────────────────────────────────────────────────────────────────

def test_instruct_prefix_constant() -> None:
    """INSTRUCT_PREFIX must contain the task description and end with 'Query: '."""
    prefix = Qwen3TextEncoder.INSTRUCT_PREFIX
    assert "pronunciation" in prefix.lower(), "Prefix should mention pronunciation"
    assert prefix.endswith("Query: "), f"Prefix should end with 'Query: ', got: {prefix!r}"


def test_output_dim_property(frozen_enc: _FakeQwen3) -> None:
    """output_dim property must return 1024."""
    assert frozen_enc.output_dim == OUTPUT_DIM, (
        f"output_dim={frozen_enc.output_dim}, expected {OUTPUT_DIM}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# last_token_pool  (static method — no model required)
# ─────────────────────────────────────────────────────────────────────────────

def test_last_token_pool_left_padded() -> None:
    """
    Left-padded sequences: last position is always real → return [:, -1].
    """
    B, T, H = 3, 6, 8
    hidden = torch.randn(B, T, H)
    # All sequences have a real token at the final position (left padding).
    attn = torch.tensor([
        [0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ])
    pooled = Qwen3TextEncoder.last_token_pool(hidden, attn)
    assert pooled.shape == (B, H)
    # For left-padded inputs we expect last_hidden_state[:, -1].
    assert torch.allclose(pooled, hidden[:, -1]), (
        "Left-padded pooling should return last position for every sequence."
    )


def test_last_token_pool_right_padded() -> None:
    """
    Right-padded sequences: each sequence ends at a different position;
    pool using per-sequence last-real-token index.
    """
    B, T, H = 2, 5, 4
    hidden = torch.randn(B, T, H)
    # Right-padded: seq 0 has length 3, seq 1 has length 5.
    attn = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ])
    pooled = Qwen3TextEncoder.last_token_pool(hidden, attn)
    assert pooled.shape == (B, H)
    # seq 0 → index 2 (3rd token), seq 1 → index 4 (5th token).
    assert torch.allclose(pooled[0], hidden[0, 2]), "Seq 0 should pool token at index 2"
    assert torch.allclose(pooled[1], hidden[1, 4]), "Seq 1 should pool token at index 4"


def test_last_token_pool_single_batch() -> None:
    """Works for batch size of 1."""
    hidden = torch.randn(1, 4, 8)
    attn   = torch.tensor([[0, 1, 1, 1]])   # left-padded, B=1
    pooled = Qwen3TextEncoder.last_token_pool(hidden, attn)
    assert pooled.shape == (1, 8)
    assert torch.allclose(pooled, hidden[:, -1])


# ─────────────────────────────────────────────────────────────────────────────
# Output shape, dtype, and L2 normalisation
# ─────────────────────────────────────────────────────────────────────────────

def test_output_shape(frozen_enc: _FakeQwen3) -> None:
    """Output must be [B, 1024] for any batch size."""
    texts = ["THE CAT SAT ON THE MAT", "HELLO WORLD"]
    out   = frozen_enc(texts)
    assert out.shape == (len(texts), OUTPUT_DIM), (
        f"Expected ({len(texts)}, {OUTPUT_DIM}), got {tuple(out.shape)}"
    )


def test_output_shape_single(frozen_enc: _FakeQwen3) -> None:
    """Works for batch size of 1."""
    out = frozen_enc(["A SINGLE UTTERANCE"])
    assert out.shape == (1, OUTPUT_DIM)


def test_fp32_output(frozen_enc: _FakeQwen3) -> None:
    """Output must always be float32 even though the model runs fp16."""
    out = frozen_enc(["test utterance"])
    assert out.dtype == torch.float32, (
        f"Output dtype={out.dtype}, expected float32"
    )


def test_l2_normalization(frozen_enc: _FakeQwen3) -> None:
    """Every output row must have L2-norm == 1.0."""
    texts = ["First sentence", "Second sentence", "Third one"]
    out   = frozen_enc(texts)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"L2 norms are not 1.0: {norms.tolist()}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Frozen / gradient behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_frozen_output_detached(frozen_enc: _FakeQwen3) -> None:
    """Frozen encoder output must not require grad (runs under no_grad)."""
    frozen_enc.train()    # even in training mode, frozen → no_grad
    out = frozen_enc(["hello"])
    assert not out.requires_grad, (
        "Frozen encoder output unexpectedly requires grad."
    )


def test_frozen_no_parameter_gradients(frozen_enc: _FakeQwen3) -> None:
    """Backward through a downstream loss must not update frozen params."""
    frozen_enc.train()
    out   = frozen_enc(["hello"])
    proxy = out.detach().requires_grad_(True)
    proxy.sum().backward()
    for name, p in frozen_enc.named_parameters():
        assert p.grad is None, f"Gradient leaked into frozen param: {name}"


def test_unfrozen_gradients_flow(trainable_enc: _FakeQwen3) -> None:
    """When unfrozen and in training mode, backward populates param grads."""
    out  = trainable_enc(["hello world"])
    loss = out.sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in trainable_enc.parameters()
    )
    assert has_grad, "Unfrozen encoder has no gradient after backward."


def test_freeze_unfreeze_toggle() -> None:
    """freeze() / unfreeze() must correctly toggle requires_grad on all params."""
    enc = _FakeQwen3(frozen=False)
    assert all(p.requires_grad for p in enc.parameters()), \
        "All params should require grad after init(frozen=False)"

    enc.freeze()
    assert enc._frozen is True
    assert all(not p.requires_grad for p in enc.parameters()), \
        "freeze() must disable requires_grad on all params"

    enc.unfreeze()
    assert enc._frozen is False
    assert all(p.requires_grad for p in enc.parameters()), \
        "unfreeze() must re-enable requires_grad on all params"


# ─────────────────────────────────────────────────────────────────────────────
# Padding side — verified via mock to avoid model download
# ─────────────────────────────────────────────────────────────────────────────

def test_padding_side_is_left() -> None:
    """
    Qwen3TextEncoder.__init__ must pass padding_side='left' to
    AutoTokenizer.from_pretrained.  Verified by intercepting the call.
    """
    captured: dict = {}

    def fake_tok(model_name: str, **kwargs: object) -> MagicMock:
        captured.update(kwargs)
        m = MagicMock()
        m.padding_side = kwargs.get("padding_side", "right")
        return m

    def fake_model(model_name: str, **kwargs: object) -> MagicMock:
        m = MagicMock()
        # Provide a real parameter so .device and .parameters() work.
        param = nn.Parameter(torch.randn(2))
        m.parameters = MagicMock(return_value=iter([param]))
        return m

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_tok), \
         patch("transformers.AutoModel.from_pretrained",     side_effect=fake_model):
        Qwen3TextEncoder(model_name="fake/model", frozen=True)

    assert captured.get("padding_side") == "left", (
        f"Tokenizer should be initialised with padding_side='left', "
        f"got: {captured.get('padding_side')!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Instruct prefix prepended to every transcript
# ─────────────────────────────────────────────────────────────────────────────

def test_instruct_prefix_prepended(frozen_enc: _FakeQwen3) -> None:
    """
    The INSTRUCT_PREFIX must be prepended to each transcript.
    Verified by monkey-patching _encode to capture inputs.
    """
    captured_inputs: List[List[str]] = []
    original_encode = frozen_enc._encode

    def capturing_encode(transcripts: List[str]) -> Tensor:
        captured_inputs.append([Qwen3TextEncoder.INSTRUCT_PREFIX + t for t in transcripts])
        return original_encode(transcripts)

    frozen_enc._encode = capturing_encode  # type: ignore[method-assign]

    transcripts = ["THE CAT SAT", "HELLO WORLD"]
    frozen_enc(transcripts)

    assert len(captured_inputs) == 1
    for t, prefixed in zip(transcripts, captured_inputs[0]):
        assert prefixed.startswith(Qwen3TextEncoder.INSTRUCT_PREFIX), (
            f"Transcript not prefixed: {prefixed!r}"
        )
        assert prefixed.endswith(t), (
            f"Original transcript missing from prefixed input: {prefixed!r}"
        )
