"""
Audio path for the pronunciation scoring architecture.

Pipeline:
    raw waveform [B, T_audio]
    → MultiResHuBERT (pretrained, output_hidden_states=True)
         → 24 hidden states, each [B, T', H]
         → learnable weighted sum              → [B, T', H]
    → Linear(H, proj_dim)                     → [B, T', proj_dim]
    → TransformerEncoder ×1 (pre-LN)          → [B, T', proj_dim]
    → return (Q for cross-attention)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .multi_res_hubert import MultiResHuBERT

log = logging.getLogger(__name__)


class AudioEncoder(nn.Module):
    """
    Full audio path: MultiResHuBERT → weighted-sum → projection → transformer.

    Args:
        cfg: Full config dict (reads model.* keys).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg     = cfg["model"]
        n_layers = mcfg["num_hidden_layers"]   # total transformer layers (f1+f2+f3)
        h_dim    = mcfg["speech_hidden_dim"]   # HuBERT hidden size (768 or 1024)
        proj_dim = mcfg["proj_dim"]            # shared projection dim

        # ── MultiResHuBERT backbone ────────────────────────────────────────
        # pretrain=True so the key layout matches the pre-training checkpoint
        # (head_hi + head_lo instead of score_head).
        self.backbone = MultiResHuBERT(
            hubert_model_name       = mcfg["hubert_model_name"],
            f1_layer_count          = mcfg["f1_layer_count"],
            f2_layer_count          = mcfg["f2_layer_count"],
            f3_layer_count          = mcfg["f3_layer_count"],
            downsample_stride       = mcfg.get("downsample_stride", 2),
            freeze_feature_extractor= mcfg.get("freeze_feature_extractor", True),
            freeze_f1_layers        = False,   # freeze handled below after load
            mask_prob               = 0.0,
            mask_length             = 0,
            num_units_hi            = mcfg.get("num_units_hi", 100),
            num_units_lo            = mcfg.get("num_units_lo", 100),
            pretrain                = True,
        )

        # ── Load pre-training checkpoint ───────────────────────────────────
        ckpt_path = mcfg.get("pretrain_checkpoint")
        if ckpt_path:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Pre-train checkpoint not found: {ckpt_path}"
                )
            log.info("Loading pre-train checkpoint: %s", ckpt_path)
            ckpt  = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("model_state", ckpt)   # handle both save formats
            missing, unexpected = self.backbone.load_state_dict(state, strict=True)
            log.info(
                "Backbone loaded (missing=%d, unexpected=%d)",
                len(missing), len(unexpected),
            )
        else:
            log.warning("No pretrain_checkpoint — backbone uses random HuBERT weights.")

        # Disable feature masking (belt-and-suspenders: also pass apply_mask=False)
        self.backbone.feat_masking.mask_prob = 0.0

        # ── Apply freeze strategy ──────────────────────────────────────────
        if mcfg.get("freeze_feature_extractor", True):
            for p in self.backbone.feature_extractor.parameters():
                p.requires_grad_(False)
            for p in self.backbone.feature_projection.parameters():
                p.requires_grad_(False)
        if mcfg.get("freeze_f1_layers", False):
            for p in self.backbone.f1_layers.parameters():
                p.requires_grad_(False)
        if mcfg.get("freeze_f2_layers", False):
            for p in self.backbone.f2_layers.parameters():
                p.requires_grad_(False)
        if mcfg.get("freeze_f3_layers", False):
            for p in self.backbone.f3_layers.parameters():
                p.requires_grad_(False)

        # ── Learnable layer-weight for weighted sum ────────────────────────
        # Initialised to uniform; softmax-normalised in forward.
        self.layer_weights = nn.Parameter(
            torch.ones(n_layers) / n_layers
        )

        # ── Projection: H_dim → proj_dim ──────────────────────────────────
        self.proj = nn.Linear(h_dim, proj_dim)

        # ── Pre-fusion transformer (pre-LN, batch_first) ──────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model   = proj_dim,
            nhead     = mcfg["pre_fusion_heads"],
            dropout   = mcfg["pre_fusion_dropout"],
            norm_first= True,       # pre-LN as in diagram
            batch_first=True,
        )
        self.pre_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = mcfg["pre_fusion_layers"],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        waveforms:      Tensor,           # [B, T_audio]
        attention_mask: Optional[Tensor], # [B, T_audio]  1=real 0=pad
    ) -> Tensor:
        """
        Args:
            waveforms:      Raw 16 kHz audio, zero-padded.   [B, T_audio]
            attention_mask: Integer mask (1=real, 0=pad).     [B, T_audio]

        Returns:
            [B, T', proj_dim] — pre-fusion audio features; Q for cross-attention.
            T' is the HuBERT feature-level length (≪ T_audio due to CNN stride).
        """
        # ── 1. Run MultiResHuBERT with all hidden states ───────────────────
        out = self.backbone(
            waveforms,
            attention_mask,
            apply_mask         = False,
            output_hidden_states = True,
        )
        all_hs = out.all_hidden_states   # List[n_layers × (B, T', H)]

        # ── 2. Learnable weighted sum over all layers ─────────────────────
        w        = torch.softmax(self.layer_weights, dim=0)   # [n_layers]
        stacked  = torch.stack(all_hs, dim=0)                 # [n_layers, B, T', H]
        features = (stacked * w[:, None, None, None]).sum(0)  # [B, T', H]

        # ── 3. Project: H → proj_dim ──────────────────────────────────────
        features = self.proj(features)                        # [B, T', proj_dim]

        # ── 4. Build padding mask for TransformerEncoder ──────────────────
        # feature mask: True = valid; PyTorch TransformerEncoder expects True = IGNORE
        T_feat    = features.shape[1]
        feat_mask = self.backbone._audio_mask_to_feat_mask(attention_mask, T_feat) \
                    if attention_mask is not None \
                    else torch.ones(features.shape[:2], dtype=torch.bool,
                                    device=features.device)
        pad_mask  = ~feat_mask   # [B, T']  True = padded position (ignore)

        # ── 5. Pre-fusion transformer ──────────────────────────────────────
        features = self.pre_transformer(
            features, src_key_padding_mask=pad_mask
        )                                                     # [B, T', proj_dim]

        return features
