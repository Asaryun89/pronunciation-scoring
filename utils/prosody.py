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
    keys = ["rms", "zcr", "peak_rate", "f0_mean", "f0_std"]
    return np.array([feats.get(k, 0.0) for k in keys], dtype=np.float32)