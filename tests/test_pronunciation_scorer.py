"""
Tests for model/pronunciation_scorer.py — PronunciationScorer.

Uses stub AudioEncoder + stub text encoder so no HF model downloads occur.
The real CrossAttentionFusion, post-fusion transformer, and MLPScoringHead
are tested end-to-end.

Updated for:
  - Upgrade 1: AudioEncoder returns List[4 × Tensor]; proj is a ModuleList.
  - Phase 0 Fix 1: model output is in (0, 10) MOS range (sigmoid × 10).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.pronunciation_scorer import PronunciationScorer

# ─────────────────────────────────────────────────────────────────────────────
# Stubs
# ─────────────────────────────────────────────────────────────────────────────

PROJ_DIM = 256
TEXT_DIM = 1024
T_FEAT   = 30
N_LAYERS = 12
H_DIM    = 768
B        = 4
T_AUDIO  = 8000


class _FakeOutput:
    def __init__(self, all_hs):
        self.all_hidden_states = all_hs


class _FakeBackbone(nn.Module):
    def __init__(self, *a, **kw) -> None:
        super().__init__()
        self.feat_masking = type("FM", (), {"mask_prob": 0.0})()
        self.feature_extractor  = nn.Identity()
        self.feature_projection = nn.Identity()
        self.f1_layers = nn.ModuleList()
        self.f2_layers = nn.ModuleList()
        self.f3_layers = nn.ModuleList()
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, wav, attn=None, apply_mask=False, output_hidden_states=False):
        B = wav.shape[0]
        all_hs = None
        if output_hidden_states:
            # Non-zero values so proj.weight receives a non-zero gradient.
            all_hs = [torch.ones(B, T_FEAT, H_DIM) for _ in range(N_LAYERS)]
        return _FakeOutput(all_hs)

    def _audio_mask_to_feat_mask(self, audio_mask, feat_len):
        return torch.ones(audio_mask.shape[0], feat_len, dtype=torch.bool)

    def load_state_dict(self, sd, strict=True): pass
    def parameters(self, recurse=True): yield self._p


class _FakeTextEncoder(nn.Module):
    def __init__(self, *a, **kw) -> None:
        super().__init__()
        self._frozen = kw.get("frozen", True)
        self._linear = nn.Linear(8, TEXT_DIM)
        if self._frozen:
            for p in self._linear.parameters():
                p.requires_grad_(False)

    def forward(self, transcripts: List[str]) -> Tensor:
        B = len(transcripts)
        out = F.normalize(self._linear(torch.randn(B, 8)).float(), p=2, dim=-1)
        return out

    def freeze(self): pass
    def unfreeze(self): pass
    def parameters(self, recurse=True): return self._linear.parameters(recurse)


def _make_cfg() -> dict:
    return {
        "model": {
            "pretrain_checkpoint":    None,
            "hubert_model_name":      "fake/model",
            "f1_layer_count":         4,
            "f2_layer_count":         4,
            "f3_layer_count":         4,
            "downsample_stride":      2,
            "freeze_feature_extractor": True,
            "freeze_f1_layers":       False,
            "freeze_f2_layers":       False,
            "freeze_f3_layers":       False,
            "num_hidden_layers":      N_LAYERS,
            "speech_hidden_dim":      H_DIM,
            "proj_dim":               PROJ_DIM,
            "pre_fusion_heads":       4,
            "pre_fusion_dropout":     0.0,
            "pre_fusion_layers":      1,
            "num_units_hi":           100,
            "num_units_lo":           100,
            "text_encoder_name":      "fake/qwen3",
            "text_encoder_dim":       TEXT_DIM,
            "freeze_text_encoder":    True,
            "text_max_length":        128,
            "fusion_heads":           4,
            "fusion_dropout":         0.0,
            "post_fusion_heads":      4,
            "post_fusion_dropout":    0.0,
            "post_fusion_layers":     2,
            "n_scores":               4,
            "scoring_dropout":        0.0,
        },
        "training": {
            "lr_speech_encoder": 5e-5,
            "lr_audio_proj":     1e-4,
            "lr_text_encoder":   0.0,
            "lr_fusion":         1e-4,
            "weight_decay":      1e-2,
            "betas":             [0.9, 0.98],
        },
    }


@pytest.fixture
def scorer() -> PronunciationScorer:
    cfg = _make_cfg()
    with patch("model.audio_encoder.MultiResHuBERT", _FakeBackbone), \
         patch("model.pronunciation_scorer.Qwen3MeanPoolEncoder", _FakeTextEncoder):
        return PronunciationScorer(cfg).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_output_shape(scorer: PronunciationScorer) -> None:
    """Full forward pass must produce [B, 4] — 4 scoring dimensions."""
    wav  = torch.randn(B, T_AUDIO)
    mask = torch.ones(B, T_AUDIO, dtype=torch.long)
    txts = ["hello"] * B
    out  = scorer(wav, mask, txts)
    assert out.shape == (B, 4), f"Expected ({B}, 4), got {tuple(out.shape)}"


def test_all_outputs_in_mos_range(scorer: PronunciationScorer) -> None:
    """Scores must be in (0, 10) MOS range (sigmoid × 10, Fix 1)."""
    wav  = torch.randn(B, T_AUDIO)
    mask = torch.ones(B, T_AUDIO, dtype=torch.long)
    txts = ["THE CAT SAT ON THE MAT"] * B
    out  = scorer(wav, mask, txts)
    assert out.min().item() > 0.0,  f"Score at or below 0: {out.min():.4f}"
    assert out.max().item() < 10.0, f"Score at or above 10: {out.max():.4f}"


def test_mos_scaling_bounded() -> None:
    """Stress test: large random inputs must still produce scores in (0, 10)."""
    cfg = _make_cfg()
    with patch("model.audio_encoder.MultiResHuBERT", _FakeBackbone), \
         patch("model.pronunciation_scorer.Qwen3MeanPoolEncoder", _FakeTextEncoder):
        model = PronunciationScorer(cfg).eval()
    for _ in range(20):
        wav  = torch.randn(2, T_AUDIO) * 10.0
        mask = torch.ones(2, T_AUDIO, dtype=torch.long)
        out  = model(wav, mask, ["x", "y"])
        assert out.min().item() > 0.0
        assert out.max().item() < 10.0


def test_gradient_flows_to_audio_proj(scorer: PronunciationScorer) -> None:
    """Backward must reach audio_encoder.proj[0].weight (ModuleList after Upgrade 1)."""
    scorer.train()
    wav  = torch.randn(2, T_AUDIO)
    mask = torch.ones(2, T_AUDIO, dtype=torch.long)
    out  = scorer(wav, mask, ["hi", "bye"])
    out.sum().backward()
    for d in range(4):
        g = scorer.audio_encoder.proj[d].weight.grad
        assert g is not None, f"audio_encoder.proj[{d}].weight has no grad"
        assert g.abs().sum() > 0, f"audio_encoder.proj[{d}].weight grad is zero"


def test_text_encoder_params_no_grad(scorer: PronunciationScorer) -> None:
    """Frozen text encoder parameters must have no grad after backward."""
    scorer.train()
    wav  = torch.randn(2, T_AUDIO)
    mask = torch.ones(2, T_AUDIO, dtype=torch.long)
    scorer(wav, mask, ["hi", "bye"]).sum().backward()
    for p in scorer.text_encoder.parameters():
        assert p.grad is None, "Frozen text encoder param has grad"


def test_batch_size_one(scorer: PronunciationScorer) -> None:
    wav  = torch.randn(1, T_AUDIO)
    mask = torch.ones(1, T_AUDIO, dtype=torch.long)
    out  = scorer(wav, mask, ["single"])
    assert out.shape == (1, 4)


def test_speech_only_ablation(scorer: PronunciationScorer) -> None:
    """speech_only=True must return different scores than speech+text."""
    wav  = torch.randn(2, T_AUDIO)
    mask = torch.ones(2, T_AUDIO, dtype=torch.long)
    txts = ["hello world", "test sentence"]

    full   = scorer(wav, mask, txts, speech_only=False)
    speech = scorer(wav, mask, txts, speech_only=True)

    # They should differ because the text path contributes to cross-attention.
    assert not torch.allclose(full, speech, atol=1e-4), \
        "speech_only and full should produce different scores"
