"""
Loss functions for Multi-resolution HuBERT pre-training.

MultiResPretrainingLoss: combined cross-entropy over masked positions for
both the high-resolution (g^q_R1, from H₃) and low-resolution (g^q_R2,
from H₂) unit-prediction heads, following the combined quantization scheme
g^q_{R1,R2}.

Fine-tuning losses (smoothed MSE + PCC) are defined inline in
training/train_scorer.py.
"""

from __future__ import annotations

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
        hi_weight: float = 1.5,
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
