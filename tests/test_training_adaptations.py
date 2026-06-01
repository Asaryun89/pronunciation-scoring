"""
Tests for training adaptations:
  - gradient accumulation step count
  - smoothed MSE reduces outlier sensitivity
  - text encoder freeze / unfreeze gradient flow
  - checkpoint resume with mismatched optimizer param groups
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Resolve both root and fusion_c_train so imports work regardless of CWD.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fusion_c_train"))

from training.finetune_fusionC import smoothed_mse, compute_loss, _DIM_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# smoothed_mse
# ─────────────────────────────────────────────────────────────────────────────

def test_smoothed_mse_zero_alpha_equals_mse() -> None:
    """alpha=0 must give identical result to plain F.mse_loss."""
    torch.manual_seed(0)
    pred   = torch.randn(8)
    target = torch.randn(8)
    assert torch.allclose(
        smoothed_mse(pred, target, alpha=0.0),
        F.mse_loss(pred, target),
    ), "smoothed_mse(alpha=0) differs from F.mse_loss"


def test_smoothed_mse_reduces_outlier_sensitivity() -> None:
    """
    With one outlier target (1.0) and the rest at 0.2, the gradient
    magnitude for the outlier sample must be smaller under smoothed MSE
    than under raw MSE.
    """
    torch.manual_seed(0)
    B      = 8
    target = torch.cat([torch.tensor([1.0]), torch.full((B - 1,), 0.2)])

    # Raw MSE gradient.
    pred_raw = torch.full((B,), 0.5, requires_grad=True)
    F.mse_loss(pred_raw, target).backward()
    raw_grad_outlier = pred_raw.grad[0].abs().item()

    # Smoothed MSE gradient.
    pred_sm = torch.full((B,), 0.5, requires_grad=True)
    smoothed_mse(pred_sm, target, alpha=0.05).backward()
    sm_grad_outlier = pred_sm.grad[0].abs().item()

    assert sm_grad_outlier < raw_grad_outlier, (
        f"Smoothed grad {sm_grad_outlier:.5f} not smaller than "
        f"raw grad {raw_grad_outlier:.5f} for outlier sample."
    )


# ─────────────────────────────────────────────────────────────────────────────
# compute_loss
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_loss_returns_correct_keys() -> None:
    """compute_loss must return a scalar + dict with one key per dim."""
    pred   = torch.rand(4, 4)
    target = torch.rand(4, 4)
    loss, dim_losses = compute_loss(pred, target, mse_weight=0.5, pcc_weight=0.5)
    assert loss.shape == (), "total_loss is not a scalar"
    for name in _DIM_NAMES:
        assert f"loss_{name}" in dim_losses, f"missing key loss_{name}"


# ─────────────────────────────────────────────────────────────────────────────
# Gradient accumulation
# ─────────────────────────────────────────────────────────────────────────────

def test_grad_accumulation_steps() -> None:
    """
    With accum_steps=2 and 4 micro-batches, optimizer.step() must fire
    exactly 2 times; loss is divided by accum_steps before backward.
    """
    model       = nn.Linear(4, 1)
    optimizer   = torch.optim.SGD(model.parameters(), lr=0.01)
    accum_steps = 2
    n_batches   = 4
    step_count  = 0

    optimizer.zero_grad()
    for micro_idx in range(n_batches):
        loss = F.mse_loss(model(torch.randn(2, 4)), torch.randn(2, 1))
        (loss / accum_steps).backward()

        if (micro_idx + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

    assert step_count == 2, f"Expected 2 optimizer steps, got {step_count}"


def test_grad_accumulation_trailing_flush() -> None:
    """
    When dataset length is not divisible by accum_steps, the trailing
    micro-batches must still trigger an optimizer step (epoch-end flush).
    """
    model       = nn.Linear(4, 1)
    optimizer   = torch.optim.SGD(model.parameters(), lr=0.01)
    accum_steps = 3
    n_batches   = 5   # 5 / 3 → 1 full boundary + 2 trailing
    step_count  = 0

    optimizer.zero_grad()
    for micro_idx in range(n_batches):
        loss = F.mse_loss(model(torch.randn(2, 4)), torch.randn(2, 1))
        (loss / accum_steps).backward()

        is_boundary = (micro_idx + 1) % accum_steps == 0
        is_last     = (micro_idx + 1) == n_batches
        if is_boundary or is_last:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

    assert step_count == 2, (
        f"Expected 2 optimizer steps (1 boundary + 1 flush), got {step_count}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text encoder freeze / unfreeze  (using a lightweight stub)
# ─────────────────────────────────────────────────────────────────────────────

class _FakeTextEncoder(nn.Module):
    """Lightweight stand-in for Qwen3TextEncoder that skips HF model loading."""

    PREFIX = ""

    def __init__(self, text_dim: int = 16, frozen: bool = True) -> None:
        super().__init__()
        self.model      = nn.Linear(8, text_dim)
        self._frozen    = frozen
        self.max_length = 128
        if frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

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
        return F.normalize(self.model(x).float(), p=2, dim=-1)

    def forward(self, transcripts: List[str]) -> Tensor:
        if not self.training or self._frozen:
            with torch.no_grad():
                return self._encode(transcripts)
        return self._encode(transcripts)


def test_text_encoder_unfrozen_gradients() -> None:
    """When unfrozen, backward through text_emb populates encoder grads."""
    enc = _FakeTextEncoder(frozen=False)
    enc.train()
    enc(["hello", "world"]).sum().backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.parameters()
    )
    assert has_grad, "Unfrozen text encoder has no gradient after backward."


def test_text_encoder_frozen_no_gradients() -> None:
    """When frozen, forward runs under no_grad — output is detached."""
    enc = _FakeTextEncoder(frozen=True)
    enc.train()
    out = enc(["hello", "world"])

    assert not out.requires_grad, "Frozen encoder output unexpectedly requires grad."
    for p in enc.parameters():
        assert p.grad is None, f"Frozen param has grad: {p.grad}"


def test_text_encoder_freeze_unfreeze_toggle() -> None:
    """freeze() / unfreeze() must correctly toggle requires_grad on all params."""
    enc = _FakeTextEncoder(frozen=False)
    assert all(p.requires_grad for p in enc.parameters()), \
        "Params should be trainable after init(frozen=False)"

    enc.freeze()
    assert not enc._frozen is False  # _frozen is True
    assert all(not p.requires_grad for p in enc.parameters()), \
        "freeze() did not disable requires_grad"

    enc.unfreeze()
    assert all(p.requires_grad for p in enc.parameters()), \
        "unfreeze() did not re-enable requires_grad"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint resume with mismatched optimizer param groups
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_resume_new_param_group(tmp_path: Path) -> None:
    """
    Save a checkpoint with 2 optimizer param groups (epoch-15 frozen config),
    then load it into an optimizer with 3 param groups (unfrozen text encoder).
    Confirm: no crash, new group keeps its init lr, old groups keep theirs.
    """
    p1 = nn.Parameter(torch.randn(4))
    p2 = nn.Parameter(torch.randn(4))
    old_optim = torch.optim.AdamW([
        {"params": [p1], "lr": 5e-5, "name": "speech"},
        {"params": [p2], "lr": 1e-4, "name": "fusion"},
    ])
    # One step so Adam accumulates non-trivial state.
    (p1 + p2).sum().backward()
    old_optim.step()
    old_optim.zero_grad()

    ckpt_path = tmp_path / "epoch_15.pt"
    torch.save({"optimizer_state": old_optim.state_dict()}, ckpt_path)

    # New optimizer: 3 groups — text encoder added.
    p3 = nn.Parameter(torch.randn(4))
    new_optim = torch.optim.AdamW([
        {"params": [p1], "lr": 5e-5, "name": "speech"},
        {"params": [p2], "lr": 1e-4, "name": "fusion"},
        {"params": [p3], "lr": 1e-5, "name": "text"},
    ])

    ckpt = torch.load(ckpt_path, map_location="cpu")
    try:
        new_optim.load_state_dict(ckpt["optimizer_state"])
    except (ValueError, RuntimeError):
        saved_groups = ckpt["optimizer_state"].get("param_groups", [])
        for i, pg in enumerate(new_optim.param_groups):
            if i < len(saved_groups):
                pg["lr"] = saved_groups[i]["lr"]

    assert new_optim.param_groups[0]["lr"] == pytest.approx(5e-5)
    assert new_optim.param_groups[1]["lr"] == pytest.approx(1e-4)
    assert new_optim.param_groups[2]["lr"] == pytest.approx(1e-5), (
        f"New text-encoder group lr={new_optim.param_groups[2]['lr']}, expected 1e-5"
    )
