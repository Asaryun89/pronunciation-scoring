"""
model/losses.py
Composite loss for CAPT pronunciation scoring (Upgrade 3).

Four components:
  DimensionHuberLoss      — per-dim Huber with annotator-variance weights
  ListMLERankingLoss      — ListMLE ranking loss targeting SRCC
  ConcordanceCorrelationLoss — CCC loss targeting PCC + mean calibration
  PronunciationScoringLoss — composite: Huber + ListMLE + CCC + optional aux CE

All losses expect pred and target in (0, 10) MOS range.
Dataset labels are in [0, 1]; multiply by 10 in the training loop before
passing here (see training/train_scorer.py).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# DimensionHuberLoss
# ─────────────────────────────────────────────────────────────────────────────

class DimensionHuberLoss(nn.Module):
    """
    Independent Huber loss per scoring dimension, weighted by inverse
    annotator variance (lambda_d) to down-weight noisy dimensions.

    Args:
        delta:    Huber transition point in MOS units (default 1.0)
        lambdas:  Per-dimension loss weights [acc, flu, pro, tot].
                  Default [1.0, 1.0, 0.5, 1.0] — lower weight on prosodic
                  (index 2) due to high inter-annotator variance.
    """

    def __init__(
        self,
        delta:   float = 1.0,
        lambdas: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.delta = delta
        if lambdas is None:
            lambdas = [1.0, 1.0, 0.5, 1.0]
        self.register_buffer(
            "lambdas", torch.tensor(lambdas, dtype=torch.float32)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: [B, 4] in (0, 10) MOS range.

        Returns:
            Scalar loss.
        """
        loss = pred.new_zeros(1)
        for d in range(4):
            loss = loss + self.lambdas[d] * F.huber_loss(
                pred[:, d], target[:, d],
                reduction="mean", delta=self.delta,
            )
        return loss / 4.0


# ─────────────────────────────────────────────────────────────────────────────
# ListMLERankingLoss
# ─────────────────────────────────────────────────────────────────────────────

class ListMLERankingLoss(nn.Module):
    """
    ListMLE ranking loss (Xia et al., 2008).

    Maximises the log-likelihood of the correct ranking (by ground-truth score,
    descending) given predictions.  Applied per dimension, averaged across
    dimensions.  Directly optimises SRCC.

    Numerically stable implementation: uses logsumexp over suffixes.
    Undefined for batch size < 2; returns 0 in that case.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: [B, 4] in (0, 10) MOS range.

        Returns:
            Scalar loss.
        """
        B = pred.shape[0]
        if B < 2:
            return pred.new_zeros(1)

        total_loss = pred.new_zeros(1)
        for d in range(4):
            _, sort_idx  = target[:, d].sort(descending=True)
            pred_sorted  = pred[:, d][sort_idx]         # [B]

            # Z-score normalise before ranking loss: prevents raw MOS scale
            # (0-10) from inflating log-sum-exp and dominating Huber.
            # Observed mean=0.475 (std 0.122) reduced to ~0.1-0.2 range.
            pred_sorted = (pred_sorted - pred_sorted.mean()) / \
                          (pred_sorted.std(unbiased=False) + 1e-8)

            loss_d = pred.new_zeros(1)
            for i in range(B):
                suffix   = pred_sorted[i:]              # [B-i]
                log_sum  = torch.logsumexp(suffix, dim=0)
                loss_d   = loss_d + (log_sum - pred_sorted[i])

            total_loss = total_loss + loss_d / B

        return total_loss / 4.0


# ─────────────────────────────────────────────────────────────────────────────
# ConcordanceCorrelationLoss
# ─────────────────────────────────────────────────────────────────────────────

class ConcordanceCorrelationLoss(nn.Module):
    """
    CCC (Lin's concordance correlation coefficient) loss.

    CCC = 2·cov(p,t) / (var(p) + var(t) + (mean(p) − mean(t))²)
    Loss = mean over dims of (1 − CCC).

    Penalises both correlation failure AND mean offset, which is critical for
    avoiding systematic bias (e.g. always predicting near the middle of the
    MOS range).  Range: [0, 2].
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: [B, 4] in (0, 10) MOS range.

        Returns:
            Scalar loss in [0, 2].
        """
        total_loss = pred.new_zeros(1)
        for d in range(4):
            p      = pred[:, d]
            t      = target[:, d]
            p_mean = p.mean()
            t_mean = t.mean()
            p_var  = p.var(unbiased=False)
            t_var  = t.var(unbiased=False)
            cov    = ((p - p_mean) * (t - t_mean)).mean()
            ccc    = (2.0 * cov) / (
                p_var + t_var + (p_mean - t_mean) ** 2 + self.eps
            )
            # CCC is mathematically in [-1, 1]; clamp for numerical stability
            # with small batches where the denominator can be near eps.
            ccc    = ccc.clamp(-1.0, 1.0)
            total_loss = total_loss + (1.0 - ccc)
        return total_loss / 4.0


# ─────────────────────────────────────────────────────────────────────────────
# PronunciationScoringLoss
# ─────────────────────────────────────────────────────────────────────────────

class PronunciationScoringLoss(nn.Module):
    """
    Composite loss for pronunciation scoring.

    L = huber
        + alpha  · listmle    (ranking, targets SRCC)
        + gamma  · ccc        (calibration, targets PCC + bias)
        + beta   · aux_ce     (optional phoneme auxiliary task)

    Warm-up schedule: for global_step < warmup_steps, alpha and gamma are
    zeroed so only Huber runs during the instability window of early training.

    Label smoothing: Gaussian noise σ=label_smooth_std MOS points is added
    to targets before any loss computation (training mode only).  Clipped to
    [0, 10].  Regularises round-number clustering in human annotations.

    Args:
        alpha:            ListMLE weight (default 0.15 — reduced from 0.30;
                          ListMLE raw mean was 2.89× Huber, causing oscillation)
        gamma:            CCC weight (default 0.35 — raised from 0.20;
                          MSE did not converge with γ=0.20, CCC std=0.175)
        beta:             Auxiliary phoneme CE weight (default 0.1)
        warmup_steps:     Steps before alpha/gamma activate (default 1000)
        huber_delta:      Huber transition point in MOS units (default 1.0)
        lambdas:          Per-dim Huber weights (default [1., 1., 0.5, 1.])
        label_smooth_std: Gaussian noise std on targets in MOS units (default 0.3)
    """

    def __init__(
        self,
        alpha:            float = 0.15,
        gamma:            float = 0.35,
        beta:             float = 0.1,
        warmup_steps:     int   = 1000,
        huber_delta:      float = 1.0,
        lambdas:          Optional[list] = None,
        label_smooth_std: float = 0.3,
    ) -> None:
        super().__init__()
        self.alpha            = alpha
        self.gamma            = gamma
        self.beta             = beta
        self.warmup_steps     = warmup_steps
        self.label_smooth_std = label_smooth_std

        self.huber   = DimensionHuberLoss(delta=huber_delta, lambdas=lambdas)
        self.listmle = ListMLERankingLoss()
        self.ccc     = ConcordanceCorrelationLoss()
        self.ce      = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        pred:            torch.Tensor,               # [B, 4] ∈ (0, 10)
        target:          torch.Tensor,               # [B, 4] ∈ (0, 10)
        global_step:     int,
        phoneme_logits:  Optional[torch.Tensor] = None,
        # [B*T', num_phonemes] or None
        phoneme_labels:  Optional[torch.Tensor] = None,
        # [B*T'] int64 (-1 = no label, ignored)
    ) -> dict:
        """
        Returns:
            dict with keys: total, huber, listmle, ccc, aux_ce.
            Each value is a scalar tensor.  Only total requires_grad=True
            (components are detached after the composite is formed).
        """
        # ── Label smoothing (training mode only) ─────────────────────────
        if self.training and self.label_smooth_std > 0:
            noise  = torch.randn_like(target) * self.label_smooth_std
            target = (target + noise).clamp(0.0, 10.0)

        # ── Warm-up gate ──────────────────────────────────────────────────
        warmed_up = (global_step >= self.warmup_steps)
        alpha_eff = self.alpha if warmed_up else 0.0
        gamma_eff = self.gamma if warmed_up else 0.0

        # ── Component losses ──────────────────────────────────────────────
        l_huber = self.huber(pred, target)

        if warmed_up:
            l_listmle = self.listmle(pred, target)
            l_ccc     = self.ccc(pred, target)
        else:
            l_listmle = pred.new_zeros(1)
            l_ccc     = pred.new_zeros(1)

        # ── Auxiliary phoneme CE ──────────────────────────────────────────
        if phoneme_logits is not None and phoneme_labels is not None:
            l_aux = self.ce(phoneme_logits, phoneme_labels)
        else:
            l_aux = pred.new_zeros(1)

        # ── Composite ─────────────────────────────────────────────────────
        total = (
            l_huber
            + alpha_eff * l_listmle
            + gamma_eff * l_ccc
            + self.beta  * l_aux
        )

        return {
            "total":   total,
            "huber":   l_huber.detach(),
            "listmle": l_listmle.detach(),
            "ccc":     l_ccc.detach(),
            "aux_ce":  l_aux.detach(),
        }
