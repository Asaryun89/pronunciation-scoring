from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class LogMelConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    add_deltas: bool = False


def compute_log_mel(audio: np.ndarray, config: LogMelConfig) -> np.ndarray:
    """Compute log-Mel spectrogram features with optional deltas."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    if not config.add_deltas:
        return log_mel

    delta = librosa.feature.delta(log_mel).astype(np.float32)
    delta2 = librosa.feature.delta(log_mel, order=2).astype(np.float32)
    features = np.concatenate([log_mel, delta, delta2], axis=0)
    return features


def stack_frames(features: np.ndarray) -> np.ndarray:
    """Return features as (frames, feat_dim)."""
    return np.transpose(features, (1, 0)).astype(np.float32)
