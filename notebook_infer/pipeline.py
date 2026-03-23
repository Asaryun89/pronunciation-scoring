"""
pipeline.py — thin wrapper around PronunciationPredictor.

Handles sys.path setup so notebooks at the repo root do not need
manual path manipulation.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import tempfile

import numpy as np

# Ensure repo root is on sys.path regardless of where the notebook was launched.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from inference.predictor import PronunciationPredictor, PredictorConfig  # noqa: E402


@dataclass
class ScoreConfig:
    """
    Configuration for a notebook inference session.

    Parameters
    ----------
    checkpoint   : path to trained HubertMultiTask .pt file
                   (None → random weights, scores are meaningless)
    device       : 'cpu' or 'cuda'
    hubert_name  : HuBERT model id or local directory path
    whisper_size : Faster-Whisper model size ('tiny', 'small', 'medium', …)
    language     : ISO-639-1 language code for ASR ('en', 'zh', …)
    text_model   : HuggingFace model name for the text embedding stream (None = disabled)
    """

    checkpoint:   Optional[str] = "ckpt_hubert_multitask/best.pt"
    device:       str           = "cpu"
    hubert_name:  str           = "facebook/hubert-base-ls960"
    whisper_size: str           = "small"
    language:     str           = "en"
    text_model:   Optional[str] = None


def load_predictor(cfg: ScoreConfig) -> PronunciationPredictor:
    """Instantiate and return a ready-to-use PronunciationPredictor."""
    pred_cfg = PredictorConfig(
        device              = cfg.device,
        hubert_name         = cfg.hubert_name,
        whisper_size        = cfg.whisper_size,
        whisper_device      = cfg.device,
        checkpoint_path     = cfg.checkpoint,
        text_model_name     = cfg.text_model,
    )
    return PronunciationPredictor(pred_cfg)


def score_file(
    wav_path:  str,
    predictor: PronunciationPredictor,
    cfg:       ScoreConfig,
) -> Dict[str, Any]:
    """
    Score pronunciation from an audio file path.

    Accepts any format supported by soundfile/librosa (wav, mp3, flac, …).
    Audio is resampled to 16 kHz internally.

    Returns
    -------
    dict with keys: total, accuracy, fluency, prosodic, completeness,
                    text, words, audio, inference_metadata
    """
    return predictor.predict(wav_path, language=cfg.language)


def score_array(
    audio:     np.ndarray,
    sr:        int,
    predictor: PronunciationPredictor,
    cfg:       ScoreConfig,
    tmp_path:  Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score pronunciation from a raw numpy float32 waveform.

    Parameters
    ----------
    audio    : float32 numpy array, shape (T,)
    sr       : sample rate of `audio`
    predictor: loaded PronunciationPredictor
    cfg      : ScoreConfig (language is read from here)
    tmp_path : temporary wav file path used internally (auto-generated if None)

    Returns
    -------
    Same dict as score_file().
    """
    import soundfile as sf

    if tmp_path is None:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    try:
        sf.write(tmp_path, audio, sr)
        result = predictor.predict(tmp_path, language=cfg.language)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    result["audio"]["path"] = "<numpy array>"
    return result
