"""
FlatHuBERT — standard HuBERT encoder without the multi-resolution block.

Architecture:
    raw waveform  [B, T_audio]
    │
    f₀  CNN Feature Extractor  →  [B, T_feat, H]   (frozen by default)
    │
    ConvFeatureMasking  →  H̃₀   [B, T_feat, H]
    │
    Positional conv + LayerNorm + Dropout
    │
    TransformerEncoder  ×N  (all layers, flat — no DOWN / UP)
    │
    UnitPredictionHead  g^q   (pre-training only)

Drop-in replacement for MultiResHuBERT in AudioEncoder.
Output format (FlatHuBERTOutput) is a superset of MultiResHuBERTOutput:
  - all_hidden_states contains one entry per transformer layer, all at [B, T_feat, H].
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.models.hubert.modeling_hubert import HubertModel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers  (shared with multi_res_hubert.py)
# ─────────────────────────────────────────────────────────────────────────────

def _feat_mask_to_additive(mask: Optional[Tensor]) -> Optional[Tensor]:
    """Bool feature mask (B, T) → additive bias (B, 1, 1, T) for HuBERT layers."""
    if mask is None:
        return None
    additive = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    additive = additive.masked_fill(~mask, torch.finfo(torch.float32).min)
    return additive[:, None, None, :]


# ─────────────────────────────────────────────────────────────────────────────
# Feature masking  (identical to MultiResHuBERT)
# ─────────────────────────────────────────────────────────────────────────────

class ConvFeatureMasking(nn.Module):
    """Randomly masks contiguous spans of CNN feature frames during pre-training."""

    def __init__(
        self,
        hidden_size: int,
        mask_prob: float = 0.065,
        mask_length: int = 10,
    ) -> None:
        super().__init__()
        self.mask_prob   = mask_prob
        self.mask_length = mask_length
        self.mask_emb    = nn.Parameter(torch.FloatTensor(hidden_size).uniform_())

    def forward(self, x: Tensor, apply_mask: bool = True) -> tuple[Tensor, Tensor]:
        B, T, H = x.shape
        mask_ids = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        if not apply_mask or not self.training:
            return x, mask_ids

        for b in range(B):
            n = max(1, int(self.mask_prob * T))
            starts = torch.randint(0, max(1, T - self.mask_length), (n,))
            for s in starts:
                mask_ids[b, s : s + self.mask_length] = True

        masked = x.clone()
        masked[mask_ids] = self.mask_emb.to(x.dtype)
        return masked, mask_ids


# ─────────────────────────────────────────────────────────────────────────────
# Unit-prediction head  (pre-training)
# ─────────────────────────────────────────────────────────────────────────────

class UnitPredictionHead(nn.Module):
    def __init__(self, hidden_size: int, num_units: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_units)

    def forward(self, x: Tensor) -> Tensor:
        """(B, T, H) → (B, T, num_units)"""
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────────────────────────────────────

class FlatHuBERTOutput(NamedTuple):
    last_hidden:       Tensor                    # (B, T_feat, H) — final layer
    logits:            Optional[Tensor]          # (B, T_feat, num_units) — pretrain only
    mask_ids:          Tensor                    # (B, T_feat) bool
    all_hidden_states: Optional[List[Tensor]] = None  # N × (B, T_feat, H)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class FlatHuBERT(nn.Module):
    """
    Standard HuBERT encoder — CNN feature extractor + flat TransformerEncoder stack.

    No downsampling / upsampling blocks.  All N transformer layers operate at
    the same temporal resolution T_feat throughout.

    Args:
        hubert_model_name:        HuggingFace model ID.
        freeze_feature_extractor: Freeze CNN f₀ weights (default True).
        mask_prob:                Feature masking probability (pre-training).
        mask_length:              Feature masking span length (pre-training).
        num_units:                Vocabulary size for the unit-prediction head.
        pretrain:                 If True, attach a unit-prediction head.
    """

    def __init__(
        self,
        hubert_model_name:        str  = "facebook/hubert-large-ll60k",
        freeze_feature_extractor: bool = True,
        mask_prob:  float = 0.065,
        mask_length: int  = 10,
        num_units:  int   = 100,
        pretrain:   bool  = False,
    ) -> None:
        super().__init__()
        self.pretrain = pretrain

        # ── Load backbone ──────────────────────────────────────────────────
        backbone: HubertModel = HubertModel.from_pretrained(hubert_model_name)
        self.hidden_size: int = backbone.config.hidden_size

        # CNN conv params for audio → feature length conversion
        self._conv_kernel: list = list(backbone.config.conv_kernel)
        self._conv_stride: list = list(backbone.config.conv_stride)

        # ── f₀: CNN feature extractor + projection ─────────────────────────
        self.feature_extractor  = backbone.feature_extractor
        self.feature_projection = backbone.feature_projection

        if freeze_feature_extractor:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)
            for p in self.feature_projection.parameters():
                p.requires_grad_(False)

        # ── Positional conv + LN (applied once before transformer stack) ────
        self.pos_conv_embed = backbone.encoder.pos_conv_embed
        self.encoder_ln     = backbone.encoder.layer_norm
        self.encoder_drop   = backbone.encoder.dropout

        # ── Flat transformer stack (all N layers, no split) ────────────────
        self.layers = nn.ModuleList(backbone.encoder.layers)

        # ── Feature masking ────────────────────────────────────────────────
        self.feat_masking = ConvFeatureMasking(
            self.hidden_size, mask_prob=mask_prob, mask_length=mask_length
        )

        # ── Output head (pre-training only) ───────────────────────────────
        if pretrain:
            self.head = UnitPredictionHead(self.hidden_size, num_units)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _audio_mask_to_feat_mask(
        self, audio_mask: Tensor, feat_len: int
    ) -> Tensor:
        """(B, T_audio) int64 mask → (B, T_feat) bool mask."""
        lengths = audio_mask.sum(dim=1).long()
        for kernel, stride in zip(self._conv_kernel, self._conv_stride):
            lengths = (lengths - kernel) // stride + 1
        lengths = lengths.clamp(min=0, max=feat_len)
        idx = torch.arange(feat_len, device=audio_mask.device).unsqueeze(0)
        return idx < lengths.unsqueeze(1)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        waveforms:             Tensor,
        attention_mask:        Optional[Tensor] = None,
        apply_mask:            bool = True,
        output_hidden_states:  bool = False,
    ) -> FlatHuBERTOutput:
        """
        Args:
            waveforms:            (B, T_audio) float32 at 16 kHz.
            attention_mask:       (B, T_audio) int64, 1=real 0=pad.
            apply_mask:           Apply feature masking (pre-training only).
            output_hidden_states: Collect each layer's output in all_hidden_states.

        Returns:
            FlatHuBERTOutput
        """
        # ── f₀: CNN ────────────────────────────────────────────────────────
        cnn_out = self.feature_extractor(waveforms).transpose(1, 2)  # (B, T', H_cnn)
        hidden  = self.feature_projection(cnn_out)                    # (B, T', H)
        T_feat  = hidden.shape[1]

        # ── Padding mask ───────────────────────────────────────────────────
        feat_mask: Optional[Tensor] = None
        if attention_mask is not None:
            feat_mask = self._audio_mask_to_feat_mask(attention_mask, T_feat)

        # ── Feature masking (H̃₀) ──────────────────────────────────────────
        hidden, mask_ids = self.feat_masking(hidden, apply_mask=apply_mask)

        # ── Positional encoding + LN ────────────────────────────────────────
        hidden = hidden + self.pos_conv_embed(hidden)
        hidden = self.encoder_ln(hidden)
        hidden = self.encoder_drop(hidden)

        # ── Flat transformer stack ─────────────────────────────────────────
        attn_mask = _feat_mask_to_additive(feat_mask)
        all_hs: Optional[List[Tensor]] = [] if output_hidden_states else None

        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attn_mask)[0]
            if all_hs is not None:
                all_hs.append(hidden)

        # ── Output head ────────────────────────────────────────────────────
        logits = self.head(hidden) if self.pretrain else None

        return FlatHuBERTOutput(
            last_hidden       = hidden,
            logits            = logits,
            mask_ids          = mask_ids,
            all_hidden_states = all_hs,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def encode(
        self,
        waveforms:      Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Inference helper — returns final hidden states (B, T', H)."""
        self.eval()
        return self.forward(
            waveforms, attention_mask, apply_mask=False
        ).last_hidden
