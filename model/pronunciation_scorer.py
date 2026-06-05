"""
Full pronunciation scoring model — wires all components together.

Forward pass:
    waveform, attention_mask, transcripts
    → AudioEncoder          → pre_fused  4 × [B, T', proj_dim]
                               [0]=accuracy  [1]=fluency
                               [2]=prosodic  [3]=total (Q for fusion)
    → Qwen3MeanPoolEncoder  → text_emb [B, text_dim]
    → CrossAttentionFusion  → fused    [B, T', proj_dim]  (uses pre_fused[3] as Q)
    → TransformerEncoder ×2 → fused    [B, T', proj_dim]
    → masked mean pool × 4  → pooled_{tot,acc,flu,pro}  [B, proj_dim]
    → 4 × _DimHead          → scores   [B, 4]  in (0, 1)

Scores are in [0, 1] to match SpeechOcean762 labels normalised to [0, 1].
Multiply by 10 for MOS display only (do not scale before computing loss).
"""

from __future__ import annotations

import logging
from typing import List

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
            model_name   = mcfg["text_encoder_name"],
            max_length   = mcfg.get("text_max_length", 128),
            frozen       = mcfg.get("freeze_text_encoder", True),
            padding_side = mcfg.get("text_padding_side", "left"),
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
            [B, 4] — scores in (0, 1).
                     Order: [total, accuracy, fluency, prosodic].
                     Multiply by 10 for MOS display only.
        """
        # ── Audio path ──────────────────────────────────────────────────────
        # Returns 4 × [B, T', proj_dim]:
        #   [0]=accuracy  [1]=fluency  [2]=prosodic  [3]=total
        pre_fused = self.audio_encoder(waveforms, attention_mask)

        # ── Text path ───────────────────────────────────────────────────────
        if speech_only:
            text_emb = torch.zeros(
                waveforms.shape[0], self._text_dim,
                device=waveforms.device, dtype=pre_fused[3].dtype,
            )
        else:
            text_emb = self.text_encoder(transcripts).to(
                device=pre_fused[3].device, dtype=pre_fused[3].dtype,
            )                                                    # [B, text_dim]

        # ── Cross-attention fusion (total stream as Q) ───────────────────────
        fused = self.fusion(pre_fused[3], text_emb)             # [B, T', proj_dim]

        # ── Post-fusion transformer ──────────────────────────────────────────
        T_feat    = fused.shape[1]
        feat_mask = self.audio_encoder.backbone._audio_mask_to_feat_mask(
            attention_mask, T_feat
        ) if attention_mask is not None else \
            torch.ones(fused.shape[:2], dtype=torch.bool, device=fused.device)
        pad_mask = ~feat_mask                                    # [B, T'] True=ignore

        fused = self.post_fusion(fused, src_key_padding_mask=pad_mask)
                                                                 # [B, T', proj_dim]

        # ── Masked mean pool ─────────────────────────────────────────────────
        m = feat_mask.unsqueeze(-1).float()                      # [B, T', 1]

        def _masked_mean(x: Tensor) -> Tensor:
            return (x * m).sum(1) / m.sum(1).clamp(min=1e-6)    # [B, proj_dim]

        pooled_tot = _masked_mean(fused)                         # post-fusion total
        pooled_acc = _masked_mean(pre_fused[0])                  # pre-fusion accuracy
        pooled_flu = _masked_mean(pre_fused[1])                  # pre-fusion fluency
        pooled_pro = _masked_mean(pre_fused[2])                  # pre-fusion prosodic

        # ── Per-dimension scoring ────────────────────────────────────────────
        score_tot = self.scoring_head.total_mlp(pooled_tot)      # [B, 1]
        score_acc = self.scoring_head.accuracy_mlp(pooled_acc)   # [B, 1]
        score_flu = self.scoring_head.fluency_mlp(pooled_flu)    # [B, 1]
        score_pro = self.scoring_head.prosodic_mlp(pooled_pro)   # [B, 1]

        # Order matches SCORE_DIMS = ["total", "accuracy", "fluency", "prosodic"]
        return torch.cat([score_tot, score_acc, score_flu, score_pro], dim=1) * 10

    # ──────────────────────────────────────────────────────────────────────
    # Optimizer
    # ──────────────────────────────────────────────────────────────────────

    def get_optimizer(self) -> AdamW:
        """
        Returns AdamW with four parameter groups:
          A — audio_encoder.backbone (MultiResHuBERT)   lr = lr_speech_encoder
          B — audio_encoder layer_weights / proj[0-3] / pre_transformer
                                                        lr = lr_audio_proj
          C — text_encoder                               lr = lr_text_encoder (0.0)
          D — fusion + post_fusion + scoring_head        lr = lr_fusion
        """
        ae = self.audio_encoder

        # Group A: backbone trainable params
        speech_params = [p for p in ae.backbone.parameters() if p.requires_grad]

        # Group B: layer_weights, 4-head projection, pre-fusion transformer
        proj_params = (
            [ae.layer_weights]
            + list(ae.proj.parameters())          # ModuleList — all 4 heads
            + list(ae.pre_transformer.parameters())
        )

        # Group C: text encoder (frozen; lr=0.0 keeps it in the param group
        # so the optimizer state is saved/restored correctly on resume)
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
