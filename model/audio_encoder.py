"""
Audio path for the pronunciation scoring architecture.

Pipeline:
    raw waveform [B, T_audio]
    → MultiResHuBERT (pretrained, output_hidden_states=True)
         → N hidden states, each [B, T', H]
         → 4 × learnable weighted sums (dim-conditioned)   → [4, B, T', H]
    → 4 × Linear(H, proj_dim)                              → 4 × [B, T', proj_dim]
    → shared TransformerEncoder ×1 (pre-LN)                → 4 × [B, T', proj_dim]
    → return list of 4 tensors

    Dimension index mapping (matches SCORE_DIMS order):
        0 → accuracy  (weighted toward H1 high-res branch, early layers)
        1 → fluency   (weighted toward H2/H3 boundary, mid-to-late layers)
        2 → prosodic  (weighted toward H3 reconstructed branch, late layers)
        3 → total     (inherits pretrained uniform weights)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .multi_res_hubert import MultiResHuBERT

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Post-load initialization helpers (called after load_state_dict)
# ─────────────────────────────────────────────────────────────────────────────

def initialize_layer_weights_from_pretrained(
    encoder: "AudioEncoder",
    existing_weights: Tensor,
    N: int,
) -> None:
    """
    Set up the [4, N] layer_weights parameter after checkpoint loading.

    existing_weights: 1D tensor [N] from a pretrained scorer ckpt (or uniform
                      torch.ones(N)/N when starting fresh).

    Row 3 (total) inherits existing_weights unchanged.
    Rows 0-2 get Gaussian peaks biased toward different MR-HuBERT layer ranges:
      - layers 0..N//3-1  ~ H1 high-res branch (fine-grained acoustic)
      - layers N//3..2N//3-1  ~ H2 low-res branch (compressed context)
      - layers 2N//3..N-1  ~ H3 reconstructed high-res branch

    Call this immediately after load_state_dict; do NOT call model.apply()
    or any function that re-randomizes backbone weights.
    """
    def _gaussian_row(center_frac: float, std_frac: float = 0.15) -> Tensor:
        centers = torch.arange(N, dtype=torch.float32)
        mu      = center_frac * N
        sigma   = std_frac * N
        g       = torch.exp(-0.5 * ((centers - mu) / sigma) ** 2)
        return g / g.sum()

    with torch.no_grad():
        w = encoder.layer_weights  # [4, N]
        w[3] = existing_weights.clone()     # total: inherit pretrained
        w[0] = _gaussian_row(0.25)          # accuracy: bias toward H1 (early)
        w[1] = _gaussian_row(0.65)          # fluency: bias toward H2/H3 boundary
        w[2] = _gaussian_row(0.85)          # prosodic: bias toward H3 (late)

    log.info("layer_weights initialized:")
    for i, name in enumerate(["accuracy", "fluency", "prosodic", "total"]):
        sm   = torch.softmax(encoder.layer_weights[i], dim=0)
        peak = sm.argmax().item()
        log.info(
            "  row %d (%s): peak at hidden state %d, top-3=%s",
            i, name, peak, sm.topk(3).indices.tolist(),
        )


def migrate_audio_proj(
    encoder: "AudioEncoder",
    old_proj_weight: Tensor,
    old_proj_bias: Tensor,
) -> None:
    """
    Warm-start all 4 projection heads from a pretrained single-head checkpoint.

    old_proj_weight: [proj_dim, h_dim] tensor from scorer ckpt key
                     ``audio_encoder.proj.weight``
    old_proj_bias:   [proj_dim] tensor from scorer ckpt key
                     ``audio_encoder.proj.bias``

    Copies to all 4 heads so early training is not destabilised by random init.
    """
    with torch.no_grad():
        for d in range(4):
            encoder.proj[d].weight.copy_(old_proj_weight)
            encoder.proj[d].bias.copy_(old_proj_bias)
    log.info("audio_proj: pretrained weights copied to all 4 projection heads.")


# ─────────────────────────────────────────────────────────────────────────────
# AudioEncoder
# ─────────────────────────────────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    """
    Full audio path: MultiResHuBERT → dim-conditioned weighted-sum →
    4 independent projections → shared pre-fusion transformer.

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
            hubert_model_name        = mcfg["hubert_model_name"],
            f1_layer_count           = mcfg["f1_layer_count"],
            f2_layer_count           = mcfg["f2_layer_count"],
            f3_layer_count           = mcfg["f3_layer_count"],
            downsample_stride        = mcfg.get("downsample_stride", 2),
            freeze_feature_extractor = mcfg.get("freeze_feature_extractor", True),
            freeze_f1_layers         = False,   # freeze handled below after load
            mask_prob                = 0.0,
            mask_length              = 0,
            num_units_hi             = mcfg.get("num_units_hi", 100),
            num_units_lo             = mcfg.get("num_units_lo", 100),
            pretrain                 = True,
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
            ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state", ckpt)

            # Strip wrapper prefixes that old training pipelines may have added.
            for prefix in ("speech_model.", "backbone.", "model."):
                if any(k.startswith(prefix) for k in state):
                    log.info("Stripping '%s' prefix from checkpoint keys.", prefix)
                    state = {k[len(prefix):]: v
                             for k, v in state.items()
                             if k.startswith(prefix)}
                    break

            # Load with strict=False so pretrain-only heads (head_hi/head_lo)
            # and any minor key-count mismatches don't crash initialisation.
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)

            # Identify which core backbone components actually loaded.
            loaded = set(state.keys()) - set(unexpected)
            core   = ["feature_extractor", "f1_layers", "f2_layers",
                      "down", "up", "f3_layers"]
            for comp in core:
                if not any(k.startswith(comp) for k in loaded):
                    log.warning(
                        "Core component '%s' NOT found in checkpoint — "
                        "weights will be random for this block.", comp
                    )

            if unexpected:
                log.info(
                    "%d checkpoint keys not used by this model (e.g. pretrain heads): "
                    "%s%s",
                    len(unexpected),
                    ", ".join(unexpected[:3]),
                    " ..." if len(unexpected) > 3 else "",
                )
            if missing:
                log.warning(
                    "%d model keys not found in checkpoint: %s%s",
                    len(missing),
                    ", ".join(missing[:3]),
                    " ..." if len(missing) > 3 else "",
                )

            log.info("Backbone loaded (missing=%d, unexpected=%d)",
                     len(missing), len(unexpected))
        else:
            log.warning("No pretrain_checkpoint — backbone uses random HuBERT weights.")

        # Disable feature masking (belt-and-suspenders; apply_mask=False in forward too)
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

        # ── Dim-conditioned layer weights [4, N] ──────────────────────────
        # Rows: [accuracy, fluency, prosodic, total]; softmax over dim=1 in forward.
        # Initialized to zeros here; immediately set to Gaussian peaks below.
        # When resuming from an old scorer checkpoint (layer_weights shape [N]),
        # train_scorer.py calls initialize_layer_weights_from_pretrained() again
        # with the old 1D weights so row 3 inherits the pretrained values.
        self.layer_weights = nn.Parameter(torch.zeros(4, n_layers))

        # ── 4 independent projection heads: H_dim → proj_dim ─────────────
        # Old key: audio_encoder.proj.{weight,bias}  (single Linear)
        # New key: audio_encoder.proj.{0,1,2,3}.{weight,bias}  (ModuleList)
        # When resuming, train_scorer.py calls migrate_audio_proj() to copy
        # the old single head's weights into all 4 heads as a warm start.
        self.proj = nn.ModuleList([nn.Linear(h_dim, proj_dim) for _ in range(4)])

        # ── Pre-fusion transformer (shared, pre-LN, batch_first) ──────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = proj_dim,
            nhead      = mcfg["pre_fusion_heads"],
            dropout    = mcfg["pre_fusion_dropout"],
            norm_first = True,       # pre-LN as in diagram
            batch_first= True,
        )
        self.pre_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = mcfg["pre_fusion_layers"],
        )

        # Default layer_weights initialization for fresh-start training.
        # When resuming from an old scorer checkpoint, train_scorer.py will
        # call initialize_layer_weights_from_pretrained() again with the
        # actual pretrained 1D weights so row 3 reflects prior training.
        _uniform = torch.ones(n_layers, dtype=torch.float32) / n_layers
        initialize_layer_weights_from_pretrained(self, _uniform, n_layers)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        waveforms:      Tensor,           # [B, T_audio]
        attention_mask: Optional[Tensor], # [B, T_audio]  1=real 0=pad
    ) -> List[Tensor]:
        """
        Args:
            waveforms:      Raw 16 kHz audio, zero-padded.   [B, T_audio]
            attention_mask: Integer mask (1=real, 0=pad).     [B, T_audio]

        Returns:
            List of 4 tensors, each [B, T', proj_dim]:
                [0] accuracy features  (Q biased toward early layers)
                [1] fluency features   (Q biased toward mid-to-late layers)
                [2] prosodic features  (Q biased toward late layers)
                [3] total features     (Q with inherited pretrained weighting)
            T' is the HuBERT feature-level length (≪ T_audio due to CNN stride).
        """
        # ── 1. Run MultiResHuBERT, collect all hidden states ───────────────
        out = self.backbone(
            waveforms,
            attention_mask,
            apply_mask           = False,
            output_hidden_states = True,
        )
        all_hs = out.all_hidden_states   # List[N × (B, T', H)]

        # ── 2. Dim-conditioned weighted sum via einsum ─────────────────────
        w       = torch.softmax(self.layer_weights, dim=1)  # [4, N]
        stacked = torch.stack(all_hs, dim=1)                # [B, N, T', H]
        # d=dim(0-3), n=hidden state, b=batch, t=time, f=feature
        dim_feats = torch.einsum("dn,bntf->dbtf", w, stacked)  # [4, B, T', H]

        # ── 3. Build padding mask for TransformerEncoder ───────────────────
        T_feat = stacked.shape[2]
        feat_mask = (
            self.backbone._audio_mask_to_feat_mask(attention_mask, T_feat)
            if attention_mask is not None
            else torch.ones(
                stacked.shape[0], T_feat,
                dtype=torch.bool, device=stacked.device,
            )
        )
        pad_mask = ~feat_mask   # [B, T']  True = padded position (ignore)

        # ── 4. Per-dimension projection + shared pre-fusion transformer ────
        pre_fused: List[Tensor] = []
        for d in range(4):
            x = self.proj[d](dim_feats[d])                           # [B, T', proj_dim]
            x = self.pre_transformer(x, src_key_padding_mask=pad_mask)
            pre_fused.append(x)

        return pre_fused   # 4 × [B, T', proj_dim]
