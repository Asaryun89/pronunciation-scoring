from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class PitchConfig:
    sample_rate: int = 16000
    hop_length: int = 160
    win_length: int = 400
    fmin: float = 50.0
    fmax: float = 500.0


def compute_pitch(audio: np.ndarray, config: PitchConfig) -> np.ndarray:
    """Compute pitch (F0) track in Hz using librosa.yin."""
    f0 = librosa.yin(
        y=audio,
        fmin=config.fmin,
        fmax=config.fmax,
        sr=config.sample_rate,
        frame_length=config.win_length,
        hop_length=config.hop_length,
    ).astype(np.float32)
    return np.nan_to_num(f0, nan=0.0)
