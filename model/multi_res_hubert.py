"""
Multi-resolution HuBERT — faithful implementation of the architecture in the paper.

Diagram flow (top → bottom)
───────────────────────────
  s  ──── raw waveform (B, T_audio)
  │
  f₀  CNN Feature Extractor  →  H̃₀  (B, T_feat, H)  [with feature masking]
  │
  f₁  High-Resolution Transformer Encoder  →  H₁  (B, T_feat, H)
  │
  DOWN  Downsampling Module  →  H₁↓  (B, T_feat//stride, H)
  │
  f₂  Low-Resolution Transformer Encoder  →  H₂  (B, T_feat//stride, H)
  │
  UP  Upsampling Module (+ skip from H₁)  →  H₂↑  (B, T_feat, H)
  │
  f₃  High-Resolution Transformer Encoder  →  H₃  (B, T_feat, H)
  │           │
  g^q_R1      g^q_R2   ← dual unit-prediction heads (pre-training)
  │           │
  g^q_{R1,R2} Quantization / combined loss

For fine-tuning on Speechocean762 the two prediction heads are replaced
with score-regression heads that operate on mean-pooled H₃ (and optionally
H₂ for the low-resolution branch).
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.models.hubert.modeling_hubert import HubertModel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _feat_mask_to_additive(mask: Optional[Tensor]) -> Optional[Tensor]:
    """
    Convert a boolean feature-level mask (B, T) where True = valid position
    to an additive attention-bias tensor (B, 1, 1, T) suitable for HuBERT
    encoder layers.

    Valid  → 0.0
    Padded → -inf   (effectively -1e9 in float16-safe form)
    """
    if mask is None:
        return None
    # mask: (B, T) bool, True = valid
    additive = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    additive = additive.masked_fill(~mask, torch.finfo(torch.float32).min)
    return additive[:, None, None, :]   # (B, 1, 1, T)


def _mean_pool(hidden: Tensor, feat_mask: Optional[Tensor]) -> Tensor:
    """
    Masked mean-pool over the time dimension.

    Args:
        hidden:    (B, T, H)
        feat_mask: (B, T) bool, True = valid.  If None, use all positions.

    Returns:
        (B, H)
    """
    if feat_mask is None:
        return hidden.mean(dim=1)
    m = feat_mask.unsqueeze(-1).float()     # (B, T, 1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


def _run_transformer_layers(
    layers: nn.ModuleList,
    hidden: Tensor,
    attn_mask: Optional[Tensor],
) -> Tensor:
    """Pass hidden states through a list of HuBERT encoder layers."""
    for layer in layers:
        hidden = layer(hidden, attention_mask=attn_mask)[0]
    return hidden


# ─────────────────────────────────────────────────────────────────────────────
# Downsampling Module  (HIGH → LOW resolution)
# ─────────────────────────────────────────────────────────────────────────────

class DownsamplingModule(nn.Module):
    """
    Reduces temporal resolution by ``stride``.

    Uses a depthwise-separable strided convolution so the operation is
    efficient and learned (rather than simple pooling).

    Input  : (B, T,  H)
    Output : (B, T', H)  where T' = ⌊(T + stride - 1) / stride⌋
    """

    def __init__(self, hidden_size: int, stride: int = 2) -> None:
        super().__init__()
        self.stride = stride
        kernel = stride * 2 - 1
        pad    = (kernel - 1) // 2
        # depthwise conv compresses time; pointwise proj mixes channels
        self.dw_conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=kernel, stride=stride, padding=pad,
            groups=hidden_size, bias=False,
        )
        self.pw_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: Tensor,
        feat_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Returns:
            h:         (B, T', H)
            mask_down: (B, T') bool or None
        """
        h = x.transpose(1, 2)                    # (B, H, T)
        h = self.act(self.pw_conv(self.dw_conv(h)))
        h = h.transpose(1, 2)                    # (B, T', H)
        h = self.norm(h)

        mask_down: Optional[Tensor] = None
        if feat_mask is not None:
            T_new = h.shape[1]
            # Max-pool the validity mask to match the conv's output length.
            # ceil_mode=True mirrors the strided conv behaviour for odd T:
            # both produce ceil(T / stride) frames.
            m = feat_mask.float().unsqueeze(1)   # (B, 1, T)
            m = F.max_pool1d(
                m, kernel_size=self.stride, stride=self.stride,
                padding=0, ceil_mode=True,
            )
            mask_down = (m.squeeze(1)[:, :T_new] > 0.5)

        return h, mask_down


# ─────────────────────────────────────────────────────────────────────────────
# Upsampling Module  (LOW → HIGH resolution, with skip connection from f₁)
# ─────────────────────────────────────────────────────────────────────────────

class UpsamplingModule(nn.Module):
    """
    Restores temporal resolution using transposed convolution, then fuses
    the result with the skip connection from the high-resolution encoder (f₁)
    through a gated linear unit.

    Inputs : h_low  (B, T', H) — output of f₂
             h_skip (B, T,  H) — output of f₁  (skip / residual)
    Output : (B, T, H)
    """

    def __init__(self, hidden_size: int, stride: int = 2) -> None:
        super().__init__()
        self.stride = stride
        self.up_conv = nn.ConvTranspose1d(
            hidden_size, hidden_size,
            kernel_size=stride, stride=stride, bias=False,
        )
        # Gated fusion: concatenate up-sampled + skip, project back to H
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.out_proj  = nn.Linear(hidden_size,     hidden_size,     bias=False)
        self.norm      = nn.LayerNorm(hidden_size)

    def forward(self, h_low: Tensor, h_skip: Tensor) -> Tensor:
        """
        Args:
            h_low:  (B, T', H) — low-resolution features from f₂
            h_skip: (B, T,  H) — high-resolution skip from f₁

        Returns:
            (B, T, H)
        """
        T_target = h_skip.shape[1]

        # Upsample with transposed convolution
        h_up = self.up_conv(h_low.transpose(1, 2)).transpose(1, 2)   # (B, ≥T, H)

        # Trim / pad to exactly T (transposed conv may be off by ±1)
        if h_up.shape[1] > T_target:
            h_up = h_up[:, :T_target, :]
        elif h_up.shape[1] < T_target:
            h_up = F.pad(h_up, (0, 0, 0, T_target - h_up.shape[1]))

        # Gated skip fusion
        combined = torch.cat([h_up, h_skip], dim=-1)   # (B, T, 2H)
        gate_in  = self.gate_proj(combined)             # (B, T, 2H)
        gate, value = gate_in.chunk(2, dim=-1)          # each (B, T, H)
        fused = torch.sigmoid(gate) * value             # (B, T, H)
        return self.norm(self.out_proj(fused))          # (B, T, H)


# ─────────────────────────────────────────────────────────────────────────────
# Feature-level masking  (applied to H̃₀ during pre-training)
# ─────────────────────────────────────────────────────────────────────────────

class ConvFeatureMasking(nn.Module):
    """
    Randomly masks contiguous spans of CNN feature frames during pre-training
    (à la HuBERT / wav2vec 2.0 masking strategy).

    During inference / fine-tuning this module is a no-op.
    """

    def __init__(
        self,
        hidden_size: int,
        mask_prob: float = 0.065,
        mask_length: int = 10,
    ) -> None:
        super().__init__()
        self.mask_prob   = mask_prob
        self.mask_length = mask_length
        # Learnable mask embedding (replaces masked frames)
        self.mask_emb = nn.Parameter(torch.FloatTensor(hidden_size).uniform_())

    def forward(self, x: Tensor, apply_mask: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:          (B, T, H)
            apply_mask: False during fine-tuning / inference.

        Returns:
            masked_x:   (B, T, H)
            mask_ids:   (B, T) bool — True where frames were masked
        """
        B, T, H = x.shape
        mask_ids = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        if not apply_mask or not self.training:
            return x, mask_ids

        for b in range(B):
            num_masks = max(1, int(self.mask_prob * T))
            starts    = torch.randint(0, max(1, T - self.mask_length), (num_masks,))
            for s in starts:
                end = min(s + self.mask_length, T)
                mask_ids[b, s:end] = True

        # Replace masked positions with the learnable embedding
        masked_x = x.clone()
        masked_x[mask_ids] = self.mask_emb.to(x.dtype)
        return masked_x, mask_ids


# ─────────────────────────────────────────────────────────────────────────────
# Unit-prediction head  (used during self-supervised pre-training)
# ─────────────────────────────────────────────────────────────────────────────

class UnitPredictionHead(nn.Module):
    """
    Projects encoder output to a logit distribution over discrete units
    (k-means cluster labels from offline clustering).

    Used for the two quantization heads g^q_R1 and g^q_R2.
    """

    def __init__(self, hidden_size: int, num_units: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_units)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, H)  →  (B, T, num_units)"""
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Score-regression head  (fine-tuning on Speechocean762)
# ─────────────────────────────────────────────────────────────────────────────

class PronunciationScoreHead(nn.Module):
    """
    Pools encoder representations and regresses to per-dimension MOS scores.

    The head consumes BOTH the high-resolution (H₃) and low-resolution (H₂)
    outputs so that coarse prosodic cues in H₂ complement fine phonetic
    detail in H₃.
    """

    def __init__(
        self,
        hidden_size: int,
        fusion_hidden: int = 512,
        n_scores: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Separate pooling projections for high-res and low-res streams
        self.hi_proj  = nn.Linear(hidden_size, fusion_hidden // 2)
        self.lo_proj  = nn.Linear(hidden_size, fusion_hidden // 2)

        self.mlp = nn.Sequential(
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.LayerNorm(fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # One independent linear head per score dimension
        self.score_heads = nn.ModuleList(
            [nn.Linear(fusion_hidden // 2, 1) for _ in range(n_scores)]
        )
        self.n_scores = n_scores

    def forward(
        self,
        h3: Tensor,
        h2: Tensor,
        feat_mask_hi: Optional[Tensor],
        feat_mask_lo: Optional[Tensor],
    ) -> Tensor:
        """
        Args:
            h3:          (B, T,  H)  — output of f₃ (high-res)
            h2:          (B, T', H)  — output of f₂ (low-res)
            feat_mask_hi: (B, T)  bool or None
            feat_mask_lo: (B, T') bool or None

        Returns:
            scores: (B, n_scores) in [0, 1]
        """
        hi_pooled = _mean_pool(h3, feat_mask_hi)   # (B, H)
        lo_pooled = _mean_pool(h2, feat_mask_lo)   # (B, H)

        fused = torch.cat(
            [self.hi_proj(hi_pooled), self.lo_proj(lo_pooled)], dim=-1
        )                                           # (B, fusion_hidden)
        fused = self.mlp(fused)                    # (B, fusion_hidden//2)

        scores = torch.cat(
            [head(fused) for head in self.score_heads], dim=-1
        )                                           # (B, n_scores)
        return torch.sigmoid(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Forward output container
# ─────────────────────────────────────────────────────────────────────────────

class MultiResHuBERTOutput(NamedTuple):
    scores:      Tensor                   # (B, n_scores) — fine-tuning
    h3:          Tensor                   # (B, T,  H)   — high-res final
    h2:          Tensor                   # (B, T', H)   — low-res branch
    logits_hi:   Optional[Tensor]         # (B, T,  V_hi) — pre-training only
    logits_lo:   Optional[Tensor]         # (B, T', V_lo) — pre-training only
    mask_ids:    Tensor                   # (B, T)  bool  — masked positions


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class MultiResHuBERT(nn.Module):
    """
    Multi-resolution HuBERT for pronunciation assessment on Speechocean762.

    The backbone follows the diagram exactly:

        f₀  →  H̃₀  →  f₁  →  DOWN  →  f₂  →  UP  →  f₃
                                 ↑skip (H₁)↑

    Two operational modes
    ─────────────────────
    • ``pretrain=True``:  returns unit-prediction logits from g^q_R1 (H₃)
                          and g^q_R2 (H₂) for self-supervised training.
    • ``pretrain=False``: returns 5-dimensional MOS scores from a
                          pronunciation-score head (fine-tuning).

    Args:
        hubert_model_name:          HuggingFace model ID.
        f1_layer_count:             Number of transformer layers in f₁.
        f2_layer_count:             Number of transformer layers in f₂.
        f3_layer_count:             Number of transformer layers in f₃.
                                    f1 + f2 + f3 must equal the total number
                                    of transformer layers in the backbone.
        downsample_stride:          Temporal stride for the DOWN module.
        freeze_feature_extractor:   Freeze CNN f₀ weights.
        freeze_f1_layers:           Freeze f₁ transformer layers.
        mask_prob:                  Feature masking probability (pre-training).
        mask_length:                Feature masking span length (pre-training).
        num_units_hi:               Vocabulary size for high-res unit head.
        num_units_lo:               Vocabulary size for low-res unit head.
        fusion_hidden:              Hidden size of the score fusion MLP.
        n_scores:                   Number of MOS score dimensions (5).
        dropout:                    Dropout in the score head.
        pretrain:                   Switch between pre-training / fine-tuning.
    """

    def __init__(
        self,
        hubert_model_name: str  = "facebook/hubert-base-ls960",
        f1_layer_count: int     = 4,
        f2_layer_count: int     = 4,
        f3_layer_count: int     = 4,
        downsample_stride: int  = 2,
        freeze_feature_extractor: bool = True,
        freeze_f1_layers: bool         = True,
        mask_prob: float   = 0.065,
        mask_length: int   = 10,
        num_units_hi: int  = 100,
        num_units_lo: int  = 50,
        fusion_hidden: int = 512,
        n_scores: int      = 5,
        dropout: float     = 0.1,
        pretrain: bool     = False,
    ) -> None:
        super().__init__()
        self.pretrain = pretrain

        # ── load pretrained backbone ───────────────────────────────────────
        backbone: HubertModel = HubertModel.from_pretrained(hubert_model_name)
        self.hidden_size: int = backbone.config.hidden_size    # 768 for base
        total_layers: int     = backbone.config.num_hidden_layers

        assert f1_layer_count + f2_layer_count + f3_layer_count == total_layers, (
            f"f1({f1_layer_count}) + f2({f2_layer_count}) + f3({f3_layer_count}) "
            f"must equal total transformer layers ({total_layers})"
        )

        # ── f₀: CNN feature extractor + projection ─────────────────────────
        self.feature_extractor = backbone.feature_extractor
        self.feature_projection = backbone.feature_projection
        # Store CNN conv parameters for audio-length → feature-length mapping.
        self._conv_kernel: list = list(backbone.config.conv_kernel)
        self._conv_stride: list = list(backbone.config.conv_stride)

        if freeze_feature_extractor:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)
            for p in self.feature_projection.parameters():
                p.requires_grad_(False)

        # Positional conv and layer-norm are applied once before f₁.
        self.pos_conv_embed = backbone.encoder.pos_conv_embed
        self.encoder_ln     = backbone.encoder.layer_norm
        self.encoder_drop   = backbone.encoder.dropout

        all_layers = backbone.encoder.layers

        # ── f₁: High-res transformer encoder ─────────────────────────────
        self.f1_layers = nn.ModuleList(all_layers[:f1_layer_count])

        if freeze_f1_layers:
            for p in self.f1_layers.parameters():
                p.requires_grad_(False)

        # ── DOWN module ───────────────────────────────────────────────────
        self.down = DownsamplingModule(self.hidden_size, stride=downsample_stride)

        # ── f₂: Low-res transformer encoder ──────────────────────────────
        self.f2_layers = nn.ModuleList(
            all_layers[f1_layer_count : f1_layer_count + f2_layer_count]
        )

        # ── UP module ─────────────────────────────────────────────────────
        self.up = UpsamplingModule(self.hidden_size, stride=downsample_stride)

        # ── f₃: High-res transformer encoder ─────────────────────────────
        self.f3_layers = nn.ModuleList(
            all_layers[f1_layer_count + f2_layer_count :]
        )

        # ── Convolutional feature masking (H̃₀) ──────────────────────────
        self.feat_masking = ConvFeatureMasking(
            self.hidden_size, mask_prob=mask_prob, mask_length=mask_length
        )

        # ── Output heads ──────────────────────────────────────────────────
        if pretrain:
            # g^q_R1 — high-res unit prediction (from H₃)
            self.head_hi = UnitPredictionHead(self.hidden_size, num_units_hi)
            # g^q_R2 — low-res unit prediction (from H₂)
            self.head_lo = UnitPredictionHead(self.hidden_size, num_units_lo)
        else:
            # Pronunciation score regression (fine-tuning)
            self.score_head = PronunciationScoreHead(
                self.hidden_size,
                fusion_hidden=fusion_hidden,
                n_scores=n_scores,
                dropout=dropout,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _audio_mask_to_feat_mask(
        self, audio_mask: Tensor, feat_len: int
    ) -> Tensor:
        """
        Convert a raw-audio attention mask (B, T_audio) → boolean feature
        mask (B, T_feat) by applying the CNN stride sequence stored at init.
        """
        lengths = audio_mask.sum(dim=1).long()   # real sample counts per item
        feat_lengths = lengths
        for kernel, stride in zip(self._conv_kernel, self._conv_stride):
            feat_lengths = (feat_lengths - kernel) // stride + 1
        feat_lengths = feat_lengths.clamp(min=0, max=feat_len)
        feat_mask = torch.arange(feat_len, device=audio_mask.device).unsqueeze(0)
        return feat_mask < feat_lengths.unsqueeze(1)   # (B, T_feat) bool

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        waveforms: Tensor,
        attention_mask: Optional[Tensor] = None,
        apply_mask: bool = True,
    ) -> MultiResHuBERTOutput:
        """
        Args:
            waveforms:      (B, T_audio) float32 at 16 kHz.
            attention_mask: (B, T_audio) int64, 1 = real sample, 0 = pad.
            apply_mask:     Whether to apply feature masking (pre-training only).

        Returns:
            MultiResHuBERTOutput — see field docs above.
        """
        # ── f₀: CNN feature extraction ────────────────────────────────────
        # HuBERT feature extractor outputs (B, H, T_feat); we need (B, T_feat, H).
        cnn_out = self.feature_extractor(waveforms).transpose(1, 2)  # (B, T_feat, H_cnn)
        hidden = self.feature_projection(cnn_out)                     # (B, T_feat, H)

        T_feat = hidden.shape[1]

        # Build feature-level boolean mask (True = valid frame).
        feat_mask_hi: Optional[Tensor] = None
        if attention_mask is not None:
            feat_mask_hi = self._audio_mask_to_feat_mask(attention_mask, T_feat)

        # ── Convolutional feature masking (H̃₀) ───────────────────────────
        hidden, mask_ids = self.feat_masking(hidden, apply_mask=apply_mask)

        # ── Positional encoding + layer-norm (shared, applied once) ───────
        hidden = hidden + self.pos_conv_embed(hidden)
        hidden = self.encoder_ln(hidden)
        hidden = self.encoder_drop(hidden)

        # ── f₁: High-resolution transformer encoder ───────────────────────
        attn_mask_hi = _feat_mask_to_additive(feat_mask_hi)
        h1 = _run_transformer_layers(self.f1_layers, hidden, attn_mask_hi)
        # h1: (B, T_feat, H)

        # ── DOWN: reduce temporal resolution ─────────────────────────────
        h1_down, feat_mask_lo = self.down(h1, feat_mask_hi)
        # h1_down: (B, T_feat//stride, H)

        # ── f₂: Low-resolution transformer encoder ───────────────────────
        attn_mask_lo = _feat_mask_to_additive(feat_mask_lo)
        h2 = _run_transformer_layers(self.f2_layers, h1_down, attn_mask_lo)
        # h2: (B, T_feat//stride, H)

        # ── UP: restore temporal resolution (+ skip from H₁) ─────────────
        h2_up = self.up(h2, h1)
        # h2_up: (B, T_feat, H)

        # ── f₃: High-resolution transformer encoder ───────────────────────
        h3 = _run_transformer_layers(self.f3_layers, h2_up, attn_mask_hi)
        # h3: (B, T_feat, H)

        # ── Output heads ──────────────────────────────────────────────────
        if self.pretrain:
            logits_hi = self.head_hi(h3)   # (B, T,  V_hi)
            logits_lo = self.head_lo(h2)   # (B, T', V_lo)
            scores    = torch.zeros(waveforms.shape[0], 0, device=waveforms.device)
        else:
            scores    = self.score_head(h3, h2, feat_mask_hi, feat_mask_lo)
            logits_hi = None
            logits_lo = None

        return MultiResHuBERTOutput(
            scores    = scores,
            h3        = h3,
            h2        = h2,
            logits_hi = logits_hi,
            logits_lo = logits_lo,
            mask_ids  = mask_ids,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def predict(
        self,
        waveforms: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Inference helper — returns denormalised MOS scores on [0, 10].
        Only valid when ``pretrain=False``.
        """
        self.eval()
        out = self.forward(waveforms, attention_mask, apply_mask=False)
        return out.scores * 10.0
