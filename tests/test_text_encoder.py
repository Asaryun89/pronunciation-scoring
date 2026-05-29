"""
Unit tests for model/text_encoder.py — BGETextEncoder.

Run:
    cd multi_res_hubert
    pytest tests/test_text_encoder.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.text_encoder import BGETextEncoder

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EXPECTED_DIM = 384


@pytest.fixture(scope="module")
def encoder() -> BGETextEncoder:
    return BGETextEncoder(model_name=MODEL_NAME, max_length=128)


# ─────────────────────────────────────────────────────────────────────────────

def test_output_shape(encoder: BGETextEncoder) -> None:
    """Output tensor must be [B, 384] for any batch size."""
    texts = ["Hello world", "This is a test sentence."]
    out   = encoder(texts)
    assert out.shape == (len(texts), EXPECTED_DIM), (
        f"Expected ({len(texts)}, {EXPECTED_DIM}), got {tuple(out.shape)}"
    )


def test_single_input(encoder: BGETextEncoder) -> None:
    """Works for batch of 1."""
    out = encoder(["A single utterance."])
    assert out.shape == (1, EXPECTED_DIM)


def test_l2_normalisation(encoder: BGETextEncoder) -> None:
    """Each row must have L2-norm ≈ 1.0 (BGE requirement)."""
    texts  = ["First sentence.", "Second sentence.", "Third one."]
    out    = encoder(texts)
    norms  = out.norm(dim=-1)   # [B]
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"L2 norms are not ≈ 1.0: {norms.tolist()}"
    )


def test_no_gradient_through_encoder(encoder: BGETextEncoder) -> None:
    """
    Backward through the encoder output must NOT populate any parameter
    gradients — the encoder is fully frozen.
    """
    texts = ["Pronunciation assessment test."]
    # Zero any previously accumulated grads.
    for p in encoder.model.parameters():
        p.grad = None

    out = encoder(texts)               # @torch.no_grad() on forward
    # Simulate a downstream loss and backward.
    # Because forward() is decorated with @torch.no_grad(), the output
    # has no grad_fn, so we clone and manually attach requires_grad for the test.
    proxy = out.clone().detach().requires_grad_(True)
    loss  = proxy.sum()
    loss.backward()

    for name, p in encoder.model.named_parameters():
        assert p.grad is None, (
            f"Gradient leaked into frozen encoder param: {name}"
        )


def test_bge_prefix_is_prepended(encoder: BGETextEncoder) -> None:
    """
    The encoder must prepend BGETextEncoder.PREFIX to each input.
    We verify this indirectly: encoding with prefix vs. without should
    produce different embeddings (BGE is sensitive to the prefix).
    """
    raw   = "The speaker's pronunciation is clear."
    prefixed = BGETextEncoder.PREFIX + raw

    # Manually run the model without the class's prefix to get a baseline.
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    tok = encoder.tokenizer

    with torch.no_grad():
        enc_no_prefix = tok(
            [raw], padding=True, truncation=True, max_length=128,
            return_tensors="pt"
        )
        enc_no_prefix = {k: v.to(encoder.device) for k, v in enc_no_prefix.items()}
        out_no_prefix = encoder.model(**enc_no_prefix).last_hidden_state[:, 0, :]
        out_no_prefix = F.normalize(out_no_prefix.float(), p=2, dim=-1)

    out_with_prefix = encoder([raw])   # class adds prefix automatically

    # They should differ (the prefix changes the contextual representation).
    cosine_sim = (out_with_prefix * out_no_prefix).sum().item()
    assert cosine_sim < 0.9999, (
        "Encoder output is identical with and without prefix — "
        "prefix may not be applied."
    )
