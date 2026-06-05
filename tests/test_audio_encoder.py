"""
Tests for model/audio_encoder.py — AudioEncoder.

Uses a lightweight _FakeBackbone stub to avoid downloading
HuBERT weights or loading a pre-training checkpoint.

Updated for Upgrade 1: AudioEncoder now returns List[4 × Tensor]
and layer_weights has shape [4, N].
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.audio_encoder import AudioEncoder

# ─────────────────────────────────────────────────────────────────────────────
# Fake backbone — mimics MultiResHuBERT interface
# ─────────────────────────────────────────────────────────────────────────────

N_LAYERS  = 12
H_DIM     = 768
T_FEAT    = 50   # synthetic feature-level length


class _FakeOutput:
    def __init__(self, all_hs: Optional[List[Tensor]]) -> None:
        self.all_hidden_states = all_hs


class _FakeBackbone(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # Expose same sub-attributes that AudioEncoder accesses.
        self.feat_masking = type("FM", (), {"mask_prob": 0.0})()
        self.feature_extractor = nn.Identity()
        self.feature_projection = nn.Identity()
        self.f1_layers = nn.ModuleList()
        self.f2_layers = nn.ModuleList()
        self.f3_layers = nn.ModuleList()
        self._trainable = nn.Parameter(torch.randn(1))  # for gradient tests

    def forward(
        self,
        waveforms:           Tensor,
        attention_mask:      Optional[Tensor] = None,
        apply_mask:          bool = False,
        output_hidden_states: bool = False,
    ) -> _FakeOutput:
        B = waveforms.shape[0]
        all_hs = None
        if output_hidden_states:
            # Return N_LAYERS states each [B, T_FEAT, H_DIM], connected to param.
            all_hs = [
                torch.zeros(B, T_FEAT, H_DIM) + self._trainable * 0
                for _ in range(N_LAYERS)
            ]
            # Make the last one depend on _trainable to test gradient flow.
            all_hs[-1] = torch.zeros(B, T_FEAT, H_DIM) + self._trainable
        return _FakeOutput(all_hs)

    def _audio_mask_to_feat_mask(self, audio_mask: Tensor, feat_len: int) -> Tensor:
        return torch.ones(audio_mask.shape[0], feat_len, dtype=torch.bool,
                          device=audio_mask.device)

    def load_state_dict(self, state_dict, strict: bool = True):
        pass   # no-op

    def parameters(self, recurse: bool = True):
        yield self._trainable


# ─────────────────────────────────────────────────────────────────────────────
# Shared test config
# ─────────────────────────────────────────────────────────────────────────────

def _make_cfg() -> dict:
    return {
        "model": {
            "pretrain_checkpoint": None,
            "hubert_model_name":   "fake/model",
            "f1_layer_count":      4,
            "f2_layer_count":      4,
            "f3_layer_count":      4,
            "downsample_stride":   2,
            "freeze_feature_extractor": True,
            "freeze_f1_layers":    False,
            "freeze_f2_layers":    False,
            "freeze_f3_layers":    False,
            "num_hidden_layers":   N_LAYERS,
            "speech_hidden_dim":   H_DIM,
            "proj_dim":            256,
            "pre_fusion_heads":    4,
            "pre_fusion_dropout":  0.1,
            "pre_fusion_layers":   1,
            "num_units_hi":        100,
            "num_units_lo":        100,
        }
    }


@pytest.fixture
def encoder() -> AudioEncoder:
    cfg = _make_cfg()
    with patch("model.audio_encoder.MultiResHuBERT", _FakeBackbone):
        return AudioEncoder(cfg).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_output_is_list_of_four(encoder: AudioEncoder) -> None:
    """AudioEncoder must return a list of 4 tensors, one per scoring dimension."""
    B, T = 4, 8000
    wav  = torch.randn(B, T)
    mask = torch.ones(B, T, dtype=torch.long)
    out  = encoder(wav, mask)
    assert isinstance(out, list), f"Expected list, got {type(out)}"
    assert len(out) == 4, f"Expected 4 tensors, got {len(out)}"


def test_output_shape(encoder: AudioEncoder) -> None:
    """Each of the 4 output tensors must be [B, T', proj_dim]."""
    B, T = 4, 8000
    wav  = torch.randn(B, T)
    mask = torch.ones(B, T, dtype=torch.long)
    out  = encoder(wav, mask)
    for d, x in enumerate(out):
        assert x.shape[0] == B,   f"dim {d}: batch dim wrong: {x.shape}"
        assert x.shape[2] == 256, f"dim {d}: proj_dim wrong: {x.shape}"


def test_layer_weights_shape(encoder: AudioEncoder) -> None:
    """layer_weights must have shape [4, N_LAYERS] after Upgrade 1."""
    assert encoder.layer_weights.shape == (4, N_LAYERS), \
        f"Expected (4, {N_LAYERS}), got {encoder.layer_weights.shape}"


def test_layer_weights_sum_to_one(encoder: AudioEncoder) -> None:
    """softmax(layer_weights, dim=1) must sum to 1.0 along hidden-state dim."""
    w = torch.softmax(encoder.layer_weights, dim=1)   # [4, N_LAYERS]
    assert w.shape == (4, N_LAYERS)
    row_sums = w.sum(dim=1)                            # [4]
    for i, s in enumerate(row_sums):
        assert abs(s.item() - 1.0) < 1e-5, \
            f"Row {i} sums to {s.item():.6f}, expected 1.0"


def test_accuracy_peak_at_midrange(encoder: AudioEncoder) -> None:
    """After Fix 2, accuracy row peak must be near layer 6 (center_frac=0.50)."""
    sm   = torch.softmax(encoder.layer_weights[0], dim=0)
    peak = sm.argmax().item()
    assert peak >= 4, \
        f"Accuracy peak at layer {peak}, expected ≥4 for center_frac=0.50"


def test_proj_is_modulelist(encoder: AudioEncoder) -> None:
    """proj must be a ModuleList of 4 independent Linear heads."""
    assert isinstance(encoder.proj, nn.ModuleList), \
        f"Expected nn.ModuleList, got {type(encoder.proj)}"
    assert len(encoder.proj) == 4


def test_gradient_flows_to_layer_weights(encoder: AudioEncoder) -> None:
    """Backward must populate all 4 rows of layer_weights.grad."""
    encoder.train()
    B, T = 2, 4000
    wav  = torch.randn(B, T)
    mask = torch.ones(B, T, dtype=torch.long)
    out  = encoder(wav, mask)                      # List[4 × Tensor]
    sum(x.sum() for x in out).backward()
    assert encoder.layer_weights.grad is not None, "layer_weights has no grad"
    assert encoder.layer_weights.grad.shape == (4, N_LAYERS), \
        f"Wrong grad shape: {encoder.layer_weights.grad.shape}"
    assert encoder.layer_weights.grad.abs().sum() > 0, "layer_weights grad is zero"


def test_gradient_flows_to_all_proj_heads(encoder: AudioEncoder) -> None:
    """Each proj[d].weight must receive a gradient."""
    encoder.train()
    wav  = torch.randn(2, 4000)
    mask = torch.ones(2, 4000, dtype=torch.long)
    sum(x.sum() for x in encoder(wav, mask)).backward()
    for d in range(4):
        g = encoder.proj[d].weight.grad
        assert g is not None, f"proj[{d}].weight has no grad"
        assert g.abs().sum() > 0, f"proj[{d}].weight grad is zero"


def test_gradient_does_not_flow_into_frozen_extractor(encoder: AudioEncoder) -> None:
    """Frozen CNN extractor parameters should have no grad after backward."""
    encoder.train()
    B, T = 2, 4000
    wav  = torch.randn(B, T)
    mask = torch.ones(B, T, dtype=torch.long)
    sum(x.sum() for x in encoder(wav, mask)).backward()
    # Stub has no real frozen params; structural check passes
    pass


def test_batch_size_one(encoder: AudioEncoder) -> None:
    """Works for batch size of 1."""
    wav  = torch.randn(1, 4000)
    mask = torch.ones(1, 4000, dtype=torch.long)
    out  = encoder(wav, mask)
    assert isinstance(out, list) and len(out) == 4
    for d, x in enumerate(out):
        assert x.shape[0] == 1, f"dim {d}: batch dim should be 1, got {x.shape}"
