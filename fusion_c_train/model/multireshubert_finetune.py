"""
Full Fusion-C fine-tuning wrapper for Multi-resolution HuBERT.

Loads a pre-trained MultiResHuBERT backbone, attaches a frozen BGETextEncoder
and a trainable FusionScoringHead, and exposes a unified forward pass:

    scores = model(waveforms, attention_mask, transcripts)   # [B, 5]

Standalone import:
    from model.multireshubert_finetune import MultiResHuBERTFinetune
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

from .multi_res_hubert import MultiResHuBERT
from .text_encoder import BGETextEncoder
from .fusion_head import FusionScoringHead

log = logging.getLogger(__name__)


def _mean_pool(hidden: Tensor, feat_mask: Optional[Tensor]) -> Tensor:
    """Masked mean-pool over time axis: (B, T, H) → (B, H)."""
    if feat_mask is None:
        return hidden.mean(dim=1)
    m = feat_mask.unsqueeze(-1).float()          # (B, T, 1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


class MultiResHuBERTFinetune(nn.Module):
    """
    Fusion-C fine-tuning wrapper.

    Speech path  : MultiResHuBERT backbone (loaded from pretrain checkpoint)
                   → mean-pool f₃ output → speech_rep [B, speech_rep_dim]
    Text path    : BGETextEncoder (frozen)
                   → [CLS] L2-normalised → text_emb [B, text_encoder_dim]
    Fusion       : FusionScoringHead (trainable)
                   → scores [B, 5] in (0, 1)

    Args:
        cfg: Parsed YAML config dict (full config, not just model section).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg = cfg["model"]
        tcfg = cfg["training"]

        # ── Speech backbone ────────────────────────────────────────────────
        # Instantiate with pretrain=True so the key set matches the checkpoint
        # exactly (head_hi + head_lo instead of score_head).
        self.speech_model = MultiResHuBERT(
            hubert_model_name       = mcfg["hubert_model_name"],
            f1_layer_count          = mcfg["f1_layer_count"],
            f2_layer_count          = mcfg["f2_layer_count"],
            f3_layer_count          = mcfg["f3_layer_count"],
            downsample_stride       = mcfg["downsample_stride"],
            freeze_feature_extractor= mcfg.get("freeze_feature_extractor", True),
            freeze_f1_layers        = mcfg.get("freeze_f1_layers", False),
            mask_prob               = 0.0,   # masking disabled for fine-tuning
            mask_length             = 0,
            num_units_hi            = mcfg.get("num_units_hi", 100),
            num_units_lo            = mcfg.get("num_units_lo", 100),
            pretrain                = True,  # matches checkpoint key layout
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
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt["model_state"]
            missing, unexpected = self.speech_model.load_state_dict(
                state, strict=True
            )
            # strict=True raises automatically; log success.
            log.info(
                "Backbone loaded (%d parameters; missing=%d, unexpected=%d)",
                sum(p.numel() for p in self.speech_model.parameters()),
                len(missing), len(unexpected),
            )
        else:
            log.warning("No pretrain_checkpoint specified — backbone is random.")

        # Disable feature masking (belt-and-suspenders: mask_prob=0 in __init__,
        # and apply_mask=False in forward; set attribute for clarity).
        self.speech_model.feat_masking.mask_prob = 0.0

        # ── Additional freeze strategy ────────────────────────────────────
        if mcfg.get("freeze_f2_layers", False):
            _freeze(self.speech_model.f2_layers)
            log.info("f2_layers frozen.")
        if mcfg.get("freeze_f3_layers", False):
            _freeze(self.speech_model.f3_layers)
            log.info("f3_layers frozen.")

        # ── Text encoder (always frozen) ──────────────────────────────────
        self.text_encoder = BGETextEncoder(
            model_name = mcfg["text_encoder_name"],
            max_length  = mcfg.get("text_max_length", 128),
        )
        _freeze(self.text_encoder)

        # ── Fusion scoring head (always trainable) ────────────────────────
        self.fusion_head = FusionScoringHead(
            speech_dim = mcfg["speech_rep_dim"],
            text_dim   = mcfg["text_encoder_dim"],
            hidden     = mcfg["fusion_hidden"],
            n_scores   = mcfg["n_scores"],
            dropout    = mcfg["dropout"],
        )

        # Store config slices for get_optimizer and forward.
        self._text_dim   = mcfg["text_encoder_dim"]
        self._lr_speech  = tcfg["lr_speech_encoder"]
        self._lr_fusion  = tcfg["lr_fusion_head"]
        self._lr_text    = tcfg.get("lr_text_encoder", 0.0)
        self._wd         = tcfg.get("weight_decay", 1e-2)
        self._betas      = tuple(tcfg.get("betas", [0.9, 0.98]))

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        waveforms:       Tensor,          # [B, T]
        attention_mask:  Optional[Tensor],# [B, T]  1=real 0=pad
        transcripts:     List[str],       # length B
        speech_only:     bool = False,    # ablation: zero-out text_emb
    ) -> Tensor:
        """
        Args:
            waveforms:      Raw 16 kHz audio, zero-padded.  [B, T]
            attention_mask: Integer mask (1=real, 0=pad).    [B, T]
            transcripts:    Utterance transcripts.           List[str]
            speech_only:    If True, replace text_emb with zeros (ablation).

        Returns:
            scores: [B, 5] in (0, 1) — [total, accuracy, fluency,
                    prosodic, completeness]
        """
        # ── Speech encoding ───────────────────────────────────────────────
        out = self.speech_model(
            waveforms, attention_mask, apply_mask=False
        )  # out.h3: [B, T_feat, H]

        # Reconstruct feature-level mask for mean-pooling.
        feat_mask: Optional[Tensor] = None
        if attention_mask is not None:
            feat_mask = self.speech_model._audio_mask_to_feat_mask(
                attention_mask, out.h3.shape[1]
            )

        speech_rep = _mean_pool(out.h3, feat_mask)  # [B, speech_rep_dim]

        # ── Text encoding ─────────────────────────────────────────────────
        if speech_only:
            text_emb = torch.zeros(
                waveforms.shape[0],
                self._text_dim,
                device=waveforms.device,
                dtype=speech_rep.dtype,
            )
        else:
            text_emb = self.text_encoder(transcripts).to(
                device=speech_rep.device, dtype=speech_rep.dtype
            )

        # ── Fusion → scores ───────────────────────────────────────────────
        return self.fusion_head(speech_rep, text_emb)   # [B, 5]

    # ──────────────────────────────────────────────────────────────────────
    # Optimizer
    # ──────────────────────────────────────────────────────────────────────

    def get_optimizer(self) -> AdamW:
        """
        Returns an AdamW with three parameter groups:
          A — speech encoder (backbone):  lr = lr_speech_encoder
          B — fusion scoring head:        lr = lr_fusion_head
          C — text encoder:               lr = lr_text_encoder  (0.0 → frozen)
        """
        speech_params = [
            p for p in self.speech_model.parameters() if p.requires_grad
        ]
        fusion_params = list(self.fusion_head.parameters())
        text_params   = list(self.text_encoder.parameters())   # all frozen

        log.info(
            "Optimizer param groups: speech=%d params | fusion=%d params | "
            "text=%d params (lr=0.0)",
            sum(p.numel() for p in speech_params),
            sum(p.numel() for p in fusion_params),
            sum(p.numel() for p in text_params),
        )

        return AdamW(
            [
                {"params": speech_params, "lr": self._lr_speech, "name": "speech"},
                {"params": fusion_params, "lr": self._lr_fusion, "name": "fusion"},
                {"params": text_params,   "lr": self._lr_text,   "name": "text"},
            ],
            weight_decay = self._wd,
            betas        = self._betas,
            eps          = 1e-6,
        )
