"""
Loss functions for Multi-resolution HuBERT.

Two modes mirror the two operating modes of MultiResHuBERT:

1. Pre-training  — MultiResPretrainingLoss
   Combines cross-entropy over masked positions for BOTH the high-resolution
   (g^q_R1, from H₃) and low-resolution (g^q_R2, from H₂) unit-prediction
   heads, following the combined quantization scheme g^q_{R1,R2}.

2. Fine-tuning   — PronunciationLoss
   Weighted MSE + Pearson-correlation loss over the five MOS score dimensions
   (accuracy, fluency, completeness, prosodic, total).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Pre-training loss  (masked unit prediction, dual heads)
# ─────────────────────────────────────────────────────────────────────────────

class MultiResPretrainingLoss(nn.Module):
    """
    Combined masked unit-prediction loss for g^q_{R1,R2}.

    For each head the loss is computed **only** on masked positions (frames
    where ``mask_ids`` is True), exactly as in standard HuBERT pre-training.

    The low-resolution mask is obtained by max-pooling the high-resolution
    mask with the same stride used by the DOWN module.

    Args:
        hi_weight:      Weight for the high-resolution (H₃) head loss.
        lo_weight:      Weight for the low-resolution  (H₂) head loss.
        downsample_stride: Stride used by the DOWN module (must match model).
    """

    def __init__(
        self,
        hi_weight: float = 1.0,
        lo_weight: float = 1.0,
        downsample_stride: int = 2,
    ) -> None:
        super().__init__()
        self.hi_weight = hi_weight
        self.lo_weight = lo_weight
        self.stride    = downsample_stride

    @staticmethod
    def masked_ce(
        logits:  Tensor,
        targets: Tensor,
        mask:    Tensor,
    ) -> Tensor:
        """
        Cross-entropy restricted to masked, non-padding positions.

        Args:
            logits:  (B, T, V)  — model logits
            targets: (B, T)     — int64 cluster labels; ``-1`` = pad (ignored)
            mask:    (B, T)     — bool; True = masked position

        Returns:
            Scalar loss (0-grad placeholder when no valid positions exist).
        """
        # Trim all three to the shortest to absorb ±1 length mismatches that
        # arise from simple T//stride target computation vs. actual CNN output.
        T = min(logits.shape[1], targets.shape[1], mask.shape[1])
        logits  = logits[:, :T]
        targets = targets[:, :T]
        mask    = mask[:, :T]

        valid = mask & (targets >= 0)
        if not valid.any():
            return logits.sum() * 0.0

        return F.cross_entropy(logits[valid], targets[valid])

    # Keep old private name as an alias so pretrain_trainer still works
    _masked_ce = staticmethod(lambda *a, **k: MultiResPretrainingLoss.masked_ce(*a, **k))

    def forward(
        self,
        logits_hi:  Tensor,
        logits_lo:  Tensor,
        targets_hi: Tensor,
        targets_lo: Tensor,
        mask_ids:   Tensor,
    ) -> Tensor:
        """
        Args:
            logits_hi:  (B, T,  V_hi) — high-res head output
            logits_lo:  (B, T', V_lo) — low-res  head output
            targets_hi: (B, T)        — cluster labels for high-res units (-1=pad)
            targets_lo: (B, T')       — cluster labels for low-res  units (-1=pad)
            mask_ids:   (B, T)  bool  — True = masked frame in H̃₀

        Returns:
            Scalar combined loss.
        """
        # ── align sequence lengths (model vs. targets may differ by ±1) ──
        T_hi = min(logits_hi.shape[1], targets_hi.shape[1], mask_ids.shape[1])
        T_lo = min(logits_lo.shape[1], targets_lo.shape[1])

        logits_hi  = logits_hi[:, :T_hi]
        targets_hi = targets_hi[:, :T_hi]
        mask_hi    = mask_ids[:, :T_hi]

        logits_lo  = logits_lo[:, :T_lo]
        targets_lo = targets_lo[:, :T_lo]

        # ── high-res loss ─────────────────────────────────────────────────
        loss_hi = self.masked_ce(logits_hi, targets_hi, mask_hi)

        # ── low-res mask: max-pool the high-res mask ───────────────────────
        mask_lo = (
            F.max_pool1d(
                mask_hi.float().unsqueeze(1),
                kernel_size=self.stride,
                stride=self.stride,
                padding=0,
                ceil_mode=True,
            )
            .squeeze(1)[:, :T_lo]
            > 0.5
        )

        loss_lo = self.masked_ce(logits_lo, targets_lo, mask_lo)

        return self.hi_weight * loss_hi + self.lo_weight * loss_lo


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning loss  (pronunciation score regression)
# ─────────────────────────────────────────────────────────────────────────────

class PronunciationLoss(nn.Module):
    """
    Weighted MSE + (1 − Pearson r) loss for multi-dimensional MOS regression.

    The Pearson term pushes the model to rank utterances correctly even when
    the absolute scale drifts; the MSE term anchors absolute values.

    Args:
        score_weights: Per-dimension loss weights
                       [accuracy, fluency, completeness, prosodic, total].
                       The ``total`` dimension is up-weighted by default.
        mse_weight:    Global coefficient for the MSE term.
        corr_weight:   Global coefficient for the (1 − Pearson r) term.
    """

    def __init__(
        self,
        score_weights: Optional[list[float]] = None,
        mse_weight: float  = 1.0,
        corr_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if score_weights is None:
            # [accuracy, fluency, completeness, prosodic, total]
            score_weights = [1.0, 1.0, 1.0, 1.0, 2.0]
        self.register_buffer(
            "score_weights",
            torch.tensor(score_weights, dtype=torch.float32),
        )
        self.mse_weight  = mse_weight
        self.corr_weight = corr_weight

    @staticmethod
    def _pearson_loss(pred: Tensor, target: Tensor) -> Tensor:
        """1 − Pearson r, averaged over score dimensions."""
        vp   = pred   - pred.mean(dim=0, keepdim=True)
        vt   = target - target.mean(dim=0, keepdim=True)
        corr = (vp * vt).sum(0) / (vp.norm(dim=0) * vt.norm(dim=0) + 1e-8)
        return (1.0 - corr).mean()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred:   (B, n_scores) in [0, 1] — model output
            target: (B, n_scores) in [0, 1] — normalised ground truth

        Returns:
            Scalar loss.
        """
        w = self.score_weights / self.score_weights.sum()

        mse_per_dim = F.mse_loss(pred, target, reduction="none").mean(0)  # (n_scores,)
        mse  = (mse_per_dim * w).sum()
        corr = self._pearson_loss(pred, target)

        return self.mse_weight * mse + self.corr_weight * corr
