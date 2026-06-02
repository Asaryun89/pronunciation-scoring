"""
Full pronunciation scoring model — wires all components together.

Forward pass:
    waveform, attention_mask, transcripts
    → AudioEncoder          → audio_q  [B, T', proj_dim]  (Q)
    → Qwen3MeanPoolEncoder  → text_emb [B, text_dim]
    → CrossAttentionFusion  → fused    [B, T', proj_dim]
    → TransformerEncoder ×2 → fused    [B, T', proj_dim]
    → mean pool             → pooled   [B, proj_dim]
    → MLPScoringHead        → scores   [B, 5]  in [0, 1]

Scores are in [0, 1]; multiply by 10 for MOS display.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW

from .audio_encoder        import AudioEncoder
from .text_encoder_qwen3   import Qwen3MeanPoolEncoder
from .cross_attention_fusion import CrossAttentionFusion
from .scoring_head         import MLPScoringHead

log = logging.getLogger(__name__)


class PronunciationScorer(nn.Module):
    """
    Multi-resolution HuBERT + Qwen3 cross-attention pronunciation scorer.

    Args:
        cfg: Full config dict.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg     = cfg["model"]
        tcfg     = cfg["training"]
        proj_dim = mcfg["proj_dim"]

        # ── Audio path ──────────────────────────────────────────────────────
        self.audio_encoder = AudioEncoder(cfg)

        # ── Text path ───────────────────────────────────────────────────────
        self.text_encoder = Qwen3MeanPoolEncoder(
            model_name    = mcfg["text_encoder_name"],
            max_length    = mcfg.get("text_max_length", 128),
            frozen        = mcfg.get("freeze_text_encoder", True),
            padding_side  = mcfg.get("text_padding_side", "left"),
        )
        self._text_dim = mcfg["text_encoder_dim"]

        # ── Cross-attention fusion ───────────────────────────────────────────
        self.fusion = CrossAttentionFusion(cfg)

        # ── Post-fusion transformer (2 layers, pre-LN) ─────────────────────
        post_layer = nn.TransformerEncoderLayer(
            d_model    = proj_dim,
            nhead      = mcfg["post_fusion_heads"],
            dropout    = mcfg["post_fusion_dropout"],
            norm_first = True,
            batch_first= True,
        )
        self.post_fusion = nn.TransformerEncoder(
            post_layer,
            num_layers = mcfg["post_fusion_layers"],
        )

        # ── Scoring head ────────────────────────────────────────────────────
        self.scoring_head = MLPScoringHead(cfg)

        # Cache LRs for get_optimizer.
        self._lr_speech = tcfg["lr_speech_encoder"]
        self._lr_proj   = tcfg["lr_audio_proj"]
        self._lr_text   = tcfg.get("lr_text_encoder", 0.0)
        self._lr_fusion = tcfg["lr_fusion"]
        self._wd        = tcfg.get("weight_decay", 1e-2)
        self._betas     = tuple(tcfg.get("betas", [0.9, 0.98]))

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        waveforms:      Tensor,       # [B, T_audio]
        attention_mask: Tensor,       # [B, T_audio]  1=real 0=pad
        transcripts:    List[str],    # length B
        speech_only:    bool = False, # ablation: zero out text contribution
    ) -> Tensor:
        """
        Args:
            waveforms:       Raw 16 kHz audio, zero-padded.    [B, T_audio]
            attention_mask:  Integer mask (1=real, 0=pad).      [B, T_audio]
            transcripts:     Utterance transcripts.             List[str]
            speech_only:     If True, replace text_emb with zeros (ablation).

        Returns:
            [B, 5] — scores in [0, 1].
                     Order: [total, accuracy, fluency, prosodic, completeness].
                     Multiply by 10 for MOS display.
        """
        # ── Audio path ──────────────────────────────────────────────────────
        audio_q = self.audio_encoder(waveforms, attention_mask)  # [B, T', proj_dim]

        # ── Text path ───────────────────────────────────────────────────────
        if speech_only:
            text_emb = torch.zeros(
                waveforms.shape[0], self._text_dim,
                device=waveforms.device, dtype=audio_q.dtype,
            )
        else:
            text_emb = self.text_encoder(transcripts).to(
                device=audio_q.device, dtype=audio_q.dtype,
            )                                                    # [B, text_dim]

        # ── Cross-attention fusion ───────────────────────────────────────────
        fused = self.fusion(audio_q, text_emb)                  # [B, T', proj_dim]

        # ── Post-fusion transformer ──────────────────────────────────────────
        # Recompute the feature-level padding mask for the post-fusion layers.
        T_feat    = fused.shape[1]
        feat_mask = self.audio_encoder.backbone._audio_mask_to_feat_mask(
            attention_mask, T_feat
        ) if attention_mask is not None else \
            torch.ones(fused.shape[:2], dtype=torch.bool, device=fused.device)
        pad_mask  = ~feat_mask                                  # [B, T'] True=ignore

        fused = self.post_fusion(fused, src_key_padding_mask=pad_mask)
                                                                # [B, T', proj_dim]

        # ── Mean pool (masked) ───────────────────────────────────────────────
        m      = feat_mask.unsqueeze(-1).float()                # [B, T', 1]
        pooled = (fused * m).sum(1) / m.sum(1).clamp(min=1e-6) # [B, proj_dim]

        # ── Score prediction ─────────────────────────────────────────────────
        return self.scoring_head(pooled)                        # [B, 5]

    # ──────────────────────────────────────────────────────────────────────
    # Optimizer
    # ──────────────────────────────────────────────────────────────────────

    def get_optimizer(self) -> AdamW:
        """
        Returns AdamW with four parameter groups:
          A — audio_encoder.backbone (MultiResHuBERT)   lr = lr_speech_encoder
          B — audio_encoder proj / layer_weights / transformer  lr = lr_audio_proj
          C — text_encoder                               lr = lr_text_encoder (0.0)
          D — fusion + post_fusion + scoring_head        lr = lr_fusion
        """
        ae = self.audio_encoder

        # Group A: backbone trainable params
        speech_params = [p for p in ae.backbone.parameters() if p.requires_grad]

        # Group B: projection, layer weights, pre-fusion transformer
        proj_params = (
            list(ae.layer_weights.unsqueeze(0))   # layer_weights is a Parameter
            + list(ae.proj.parameters())
            + list(ae.pre_transformer.parameters())
        )
        # Clean: use sets to avoid duplicates (layer_weights is already a leaf)
        proj_params = [ae.layer_weights] + list(ae.proj.parameters()) + \
                      list(ae.pre_transformer.parameters())

        # Group C: text encoder
        text_params = list(self.text_encoder.parameters())

        # Group D: fusion modules + scoring head
        fusion_params = (
            list(self.fusion.parameters())
            + list(self.post_fusion.parameters())
            + list(self.scoring_head.parameters())
        )

        log.info(
            "Optimizer groups: speech=%d | proj=%d | text=%d (lr=%.0e) | fusion=%d",
            sum(p.numel() for p in speech_params),
            sum(p.numel() for p in proj_params),
            sum(p.numel() for p in text_params),
            self._lr_text,
            sum(p.numel() for p in fusion_params),
        )

        return AdamW(
            [
                {"params": speech_params, "lr": self._lr_speech, "name": "speech"},
                {"params": proj_params,   "lr": self._lr_proj,   "name": "audio_proj"},
                {"params": text_params,   "lr": self._lr_text,   "name": "text"},
                {"params": fusion_params, "lr": self._lr_fusion, "name": "fusion"},
            ],
            weight_decay = self._wd,
            betas        = self._betas,
            eps          = 1e-6,
        )
