from __future__ import annotations

from typing import List


def uniform_alignment(num_frames: int, num_phonemes: int) -> List[int]:
    """Split frames uniformly across phonemes for a simple forced alignment stub."""
    if num_phonemes <= 1:
        return [num_frames]
    base = num_frames // num_phonemes
    rem = num_frames % num_phonemes
    lengths = [base + (1 if i < rem else 0) for i in range(num_phonemes)]
    return lengths
