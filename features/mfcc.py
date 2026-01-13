from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import librosa
import numpy as np


@dataclass
class MFCCConfig:
    sample_rate: int = 16000
    n_mfcc: int = 13
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    add_deltas: bool = True


def compute_mfcc(audio: np.ndarray, config: MFCCConfig) -> np.ndarray:
    """Compute MFCC features with optional delta and delta-delta."""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
    ).astype(np.float32)

    if not config.add_deltas:
        return mfcc

    delta = librosa.feature.delta(mfcc).astype(np.float32)
    delta2 = librosa.feature.delta(mfcc, order=2).astype(np.float32)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return features


def stack_frames(features: np.ndarray) -> np.ndarray:
    """Return features as (frames, feat_dim)."""
    return np.transpose(features, (1, 0)).astype(np.float32)
