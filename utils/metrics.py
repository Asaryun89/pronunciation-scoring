from __future__ import annotations

import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error."""
    return torch.mean(torch.abs(pred - target)).item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error."""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()
