from __future__ import annotations

from typing import List, Tuple

import librosa
import numpy as np


def uniform_alignment(num_frames: int, num_phonemes: int) -> List[int]:
    """Split frames uniformly across phonemes for a simple forced alignment stub."""
    if num_phonemes <= 1:
        return [num_frames]
    base = num_frames // num_phonemes
    rem = num_frames % num_phonemes
    lengths = [base + (1 if i < rem else 0) for i in range(num_phonemes)]
    return lengths


def dtw_alignment(ref_feats: np.ndarray, learner_feats: np.ndarray) -> List[Tuple[int, int]]:
    """Align reference and learner features using DTW and return frame index pairs."""
    if ref_feats.size == 0 or learner_feats.size == 0:
        return []
    _, path = librosa.sequence.dtw(X=ref_feats.T, Y=learner_feats.T, metric="euclidean")
    path = path[::-1]
    return [(int(i), int(j)) for i, j in path]


def uniform_frame_alignment(num_ref_frames: int, num_learner_frames: int) -> List[Tuple[int, int]]:
    """Align frames linearly as a lightweight forced alignment placeholder."""
    if num_ref_frames <= 0 or num_learner_frames <= 0:
        return []
    if num_ref_frames == 1:
        return [(0, min(0, num_learner_frames - 1))]
    path = []
    for i in range(num_ref_frames):
        j = int(round(i * (num_learner_frames - 1) / max(num_ref_frames - 1, 1)))
        j = min(max(j, 0), num_learner_frames - 1)
        path.append((i, j))
    return path
