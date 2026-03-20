import numpy as np

def basic_prosody_features(audio: np.ndarray, sr: int = 16000) -> dict:
    """
    Lightweight prosody feature set.
    If pyworld is installed, use it for F0 extraction. Otherwise fallback.
    """
    feats = {}

    # Energy features
    feats["rms"] = float(np.sqrt(np.mean(audio**2) + 1e-8))
    feats["zcr"] = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)

    # Speaking rate proxy: number of peaks above threshold per second
    thr = 0.2 * (np.max(np.abs(audio)) + 1e-8)
    peaks = np.sum((np.abs(audio[1:-1]) > thr) &
                   (np.abs(audio[1:-1]) > np.abs(audio[:-2])) &
                   (np.abs(audio[1:-1]) > np.abs(audio[2:])))
    duration = len(audio) / sr
    feats["peak_rate"] = float(peaks / max(duration, 1e-6))

    # F0 (optional)
    try:
        import pyworld as pw
        f0, t = pw.dio(audio.astype(np.float64), sr)
        f0 = pw.stonemask(audio.astype(np.float64), f0, t, sr)
        f0 = f0[f0 > 0]
        if len(f0) > 5:
            feats["f0_mean"] = float(np.mean(f0))
            feats["f0_std"] = float(np.std(f0))
        else:
            feats["f0_mean"] = 0.0
            feats["f0_std"] = 0.0
    except Exception:
        feats["f0_mean"] = 0.0
        feats["f0_std"] = 0.0

    return feats

def prosody_to_vector(feats: dict) -> np.ndarray:
    """
    Return a normalized 5-dim vector in approximately [0, 1].

    Normalization constants are fixed (not dataset-dependent) so training and
    inference produce the same values without a pre-computed mean/std:
      rms        — already ~[0, 1] after peak-normalization of audio
      zcr        — already ~[0, 0.5] for speech
      peak_rate  — log1p then divide by 8.5  (covers up to ~5000 peaks/sec)
      f0_mean    — divide by 400.0  (Hz; covers 0 and 80–400 Hz range)
      f0_std     — divide by 80.0   (Hz; typical std is 0–80 Hz)
    """
    rms       = float(feats.get("rms",       0.0))
    zcr       = float(feats.get("zcr",       0.0))
    peak_rate = float(np.log1p(feats.get("peak_rate", 0.0)) / 8.5)
    f0_mean   = float(feats.get("f0_mean",   0.0) / 400.0)
    f0_std    = float(feats.get("f0_std",    0.0) / 80.0)
    return np.array([rms, zcr, peak_rate, f0_mean, f0_std], dtype=np.float32)