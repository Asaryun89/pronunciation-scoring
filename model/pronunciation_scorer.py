"""
Full pronunciation scoring model — wires all components together.

Forward pass (Upgrade 2):
    waveform, attention_mask, transcripts
    → AudioEncoder          → pre_fused  4 × [B, T', proj_dim]
                               [0]=accuracy  [1]=fluency
                               [2]=prosodic  [3]=total (Q for fusion)
    → TokenTextProjection   → text_feats [B, N, proj_dim]
                               text_pad   [B, N] bool (True = pad)
    → CrossAttentionFusion  → fused      [B, T', proj_dim]
                               (Q=pre_fused[3], K/V=text_feats[all N tokens])
    → TransformerEncoder ×2 → fused      [B, T', proj_dim]
    → masked mean pool × 4  → pooled_{tot,acc,flu,pro}  [B, proj_dim]
    → 4 × _DimHead          → scores     [B, 4]  in (0, 1) × 10 = (0, 10) MOS

Scores are in (0, 10) to match SpeechOcean762 raw MOS range.
Dataset labels are pre-normalised to [0, 1]; multiply labels by 10 before
passing to PronunciationScoringLoss (see training/train_scorer.py).
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW

from .audio_encoder          import AudioEncoder
from .text_encoder_qwen3     import Qwen3MeanPoolEncoder   # used only as a fallback type alias
from .text_projection        import TokenTextProjection
from .cross_attention_fusion import CrossAttentionFusion
from .scoring_head           import MLPScoringHead

log = logging.getLogger(__name__)


class PronunciationScorer(nn.Module):
    """
    Multi-resolution HuBERT + Qwen3 cross-attention pronunciation scorer.

    Key attributes (for validation / gradient checks):
        audio_encoder.layer_weights  [4, 12]
        audio_encoder.proj           ModuleList of 4 Linear(768, 256)
        text_projection              TokenTextProjection (forward → seq + mask)
        text_projection.proj         Linear(1024, 256)  — trainable
        cross_attn_fusion            CrossAttentionFusion
        cross_attn_fusion.mha        nn.MultiheadAttention

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

        # ── Text path (Upgrade 2) ────────────────────────────────────────────
        # TokenTextProjection wraps frozen Qwen3 + trainable Linear+LN.
        # Returns (token_seq [B,N,256], key_padding_mask [B,N]).
        self.text_projection = TokenTextProjection(cfg)
        self._proj_dim = proj_dim   # used in speech_only zero-tensor

        # ── Cross-attention fusion (Upgrade 2) ───────────────────────────────
        # Accepts per-token K/V sequence; renamed from self.fusion.
        self.cross_attn_fusion = CrossAttentionFusion(cfg)

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

    # ── Backward-compat alias ─────────────────────────────────────────────
    # Some existing test code and evaluation scripts access self.text_encoder.
    # Return the text_projection module so they still get parameters/device.
    @property
    def text_encoder(self):
        return self.text_projection

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
            speech_only:     If True, replace text with a single zero token.

        Returns:
            [B, 4] — scores in (0, 10) MOS range.
                     Order: [total, accuracy, fluency, prosodic].
        """
        # ── Audio path ──────────────────────────────────────────────────────
        # Returns 4 × [B, T', proj_dim]:
        #   [0]=accuracy  [1]=fluency  [2]=prosodic  [3]=total
        pre_fused = self.audio_encoder(waveforms, attention_mask)
        B, T_feat = pre_fused[3].shape[0], pre_fused[3].shape[1]
        dtype     = pre_fused[3].dtype
        device    = pre_fused[3].device

        # ── Text path (Upgrade 2: per-token sequence) ────────────────────────
        if speech_only:
            # Single zero token — cross-attention will attend to nothing.
            text_feats   = torch.zeros(B, 1, self._proj_dim,
                                        device=device, dtype=dtype)
            text_pad_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        else:
            text_feats, text_pad_mask = self.text_projection(transcripts)
            text_feats = text_feats.to(device=device, dtype=dtype)
            text_pad_mask = text_pad_mask.to(device=device)

        # ── Cross-attention fusion (total stream as Q) ───────────────────────
        fused = self.cross_attn_fusion(
            query            = pre_fused[3],
            key_value        = text_feats,
            key_padding_mask = text_pad_mask,
        )                                                    # [B, T', proj_dim]

        # ── Post-fusion transformer ──────────────────────────────────────────
        feat_mask = self.audio_encoder.backbone._audio_mask_to_feat_mask(
            attention_mask, T_feat
        ) if attention_mask is not None else \
            torch.ones(B, T_feat, dtype=torch.bool, device=device)
        pad_mask = ~feat_mask                                # [B, T'] True=ignore

        fused = self.post_fusion(fused, src_key_padding_mask=pad_mask)

        # ── Masked mean pool ─────────────────────────────────────────────────
        m = feat_mask.unsqueeze(-1).float()                  # [B, T', 1]

        def _masked_mean(x: Tensor) -> Tensor:
            return (x * m).sum(1) / m.sum(1).clamp(min=1e-6)

        pooled_tot = _masked_mean(fused)                     # post-fusion total
        pooled_acc = _masked_mean(pre_fused[0])
        pooled_flu = _masked_mean(pre_fused[1])
        pooled_pro = _masked_mean(pre_fused[2])

        # TODO Upgrade 3 aux: add phoneme head when frame labels available
        # phoneme_logits = None; phoneme_labels = None

        # ── Per-dimension scoring ────────────────────────────────────────────
        score_tot = self.scoring_head.total_mlp(pooled_tot)
        score_acc = self.scoring_head.accuracy_mlp(pooled_acc)
        score_flu = self.scoring_head.fluency_mlp(pooled_flu)
        score_pro = self.scoring_head.prosodic_mlp(pooled_pro)

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
          C — text_projection.text_model (Qwen3, frozen) lr = lr_text_encoder (0)
          D — text_projection.proj/norm + cross_attn_fusion + post_fusion
              + scoring_head                            lr = lr_fusion
        """
        ae = self.audio_encoder

        # Group A: backbone trainable params
        speech_params = [p for p in ae.backbone.parameters() if p.requires_grad]

        # Group B: audio encoder (layer_weights, 4-head proj, pre-transformer)
        proj_params = (
            [ae.layer_weights]
            + list(ae.proj.parameters())
            + list(ae.pre_transformer.parameters())
        )

        # Group C: frozen Qwen3 LM backbone (lr=0 keeps optimizer state intact)
        text_params = list(self.text_projection.text_model.parameters())

        # Group D: trainable text proj head + fusion + scoring
        fusion_params = (
            list(self.text_projection.proj.parameters())
            + list(self.text_projection.norm.parameters())
            + list(self.cross_attn_fusion.parameters())
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
