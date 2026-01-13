from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import librosa
import numpy as np


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    max_audio_seconds: float = 10.0
    top_db: int = 30


def load_wav(path: str, sample_rate: int) -> np.ndarray:
    """Load a WAV file as a mono float32 numpy array."""
    audio, _sr = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def normalize(audio: np.ndarray) -> np.ndarray:
    """Peak normalize to [-1, 1] range."""
    peak = np.max(np.abs(audio)) + 1e-9
    return (audio / peak).astype(np.float32)


def trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Trim leading/trailing silence using librosa."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed.astype(np.float32)


def pad_or_trim(audio: np.ndarray, sample_rate: int, max_seconds: float) -> np.ndarray:
    """Pad or trim audio to max_seconds."""
    target_len = int(sample_rate * max_seconds)
    if audio.shape[0] >= target_len:
        return audio[:target_len].astype(np.float32)
    pad_width = target_len - audio.shape[0]
    return np.pad(audio, (0, pad_width)).astype(np.float32)


def preprocess_audio(path: str, config: AudioConfig) -> np.ndarray:
    """Load and preprocess audio for feature extraction."""
    audio = load_wav(path, config.sample_rate)
    audio = trim_silence(audio, top_db=config.top_db)
    audio = normalize(audio)
    audio = pad_or_trim(audio, config.sample_rate, config.max_audio_seconds)
    return audio


def dummy_audio(sample_rate: int, seconds: float) -> np.ndarray:
    """Generate dummy random audio for example usage."""
    num_samples = int(sample_rate * seconds)
    return np.random.randn(num_samples).astype(np.float32)
