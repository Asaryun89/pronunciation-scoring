"""
Contrastive Ordinal (CONO) regularizer for pronunciation scoring features.

Forces feature-space distances to reflect score ordering:
  diversity term  — repel score-group centroids proportional to ordinal distance
  tightness term  — pull samples toward their score-group centroid

Reference: inspired by HiPPO (Yan et al., 2025).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CONORegularizer(nn.Module):
    """
    Contrastive Ordinal regularizer.

    Applied to per-dimension feature vectors and the corresponding
    normalised scores.  Speechocean762 scores are integers [0, 10]
    mapped to [0, 1], so we round to 1 d.p. to create discrete groups.

    Args:
        lambda_d: Weight for diversity (inter-group repulsion) term.
        lambda_t: Weight for tightness (intra-group compactness) term.

    Forward:
        features: [B, D] — hidden representation for one score dimension
        scores:   [B]    — normalised scores in [0, 1]
        returns:  scalar Tensor
    """

    def __init__(self, lambda_d: float = 1.0, lambda_t: float = 1.0) -> None:
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t

    def forward(self, features: Tensor, scores: Tensor) -> Tensor:
        # Bin continuous scores to discrete groups (round to nearest 0.1).
        score_groups = (scores * 10.0).round() / 10.0

        unique = score_groups.unique()
        if unique.numel() <= 1:
            # Only one score group in batch — loss is 0 (no pairs to repel).
            return features.sum() * 0.0

        # ── Centroids ─────────────────────────────────────────────────────
        centroids = {}
        for s in unique:
            mask = score_groups == s
            centroids[s.item()] = features[mask].mean(dim=0)

        unique_vals = list(centroids.keys())
        n_unique    = len(unique_vals)

        # ── Diversity (inter-group repulsion weighted by ordinal distance) ─
        l_div   = features.sum() * 0.0
        n_pairs = 0
        for i in range(n_unique):
            for j in range(i + 1, n_unique):
                si, sj  = unique_vals[i], unique_vals[j]
                dist    = (centroids[si] - centroids[sj]).pow(2).sum()
                penalty = abs(si - sj)         # ordinal weight
                l_div   = l_div - penalty * dist   # negative → minimise = push apart
                n_pairs += 1
        if n_pairs > 0:
            l_div = l_div / n_pairs

        # ── Tightness (intra-group compactness) ───────────────────────────
        l_tight = features.sum() * 0.0
        for s_val, centroid in centroids.items():
            group   = features[score_groups == s_val]
            l_tight = l_tight + (group - centroid).pow(2).sum(dim=1).mean()
        l_tight = l_tight / n_unique

        return self.lambda_d * l_div + self.lambda_t * l_tight
