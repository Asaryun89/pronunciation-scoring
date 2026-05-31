"""
Step-based learning-rate schedule: linear warmup → cosine decay.

This is the standard schedule used in HuBERT / wav2vec 2.0 pre-training.
Unlike epoch-based schedulers it plays nicely with gradient accumulation
and allows warming up over a fixed number of steps regardless of dataset size.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_warmup_cosine_schedule(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Returns a LambdaLR scheduler that implements:

      step ∈ [0, warmup_steps)      → lr = peak_lr × (step / warmup_steps)
      step ∈ [warmup_steps, total)  → lr = peak_lr × ½(1 + cos(π·progress))
                                       clipped at min_lr_ratio × peak_lr

    Args:
        optimizer:     The optimizer whose lr groups will be scaled.
        warmup_steps:  Number of linear-warmup steps (e.g. 10 000).
        total_steps:   Total number of training steps.
        min_lr_ratio:  Floor for the cosine tail (0 = decay to zero).

    Returns:
        A ``LambdaLR`` scheduler.  Call ``scheduler.step()`` once per
        *optimizer step* (not per epoch).
    """

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        decay_steps = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / decay_steps
        cosine   = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, _lr_lambda)
