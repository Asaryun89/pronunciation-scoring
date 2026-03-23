import numpy as np
import soundfile as sf
import scipy.signal as sps
try:
    import webrtcvad
except ImportError:  # pragma: no cover - depends on runtime environment
    webrtcvad = None

def read_wav(path: str):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr

def resample(audio: np.ndarray, sr_in: int, sr_out: int = 16000) -> tuple[np.ndarray, int]:
    if sr_in == sr_out:
        return audio, sr_in
    gcd = np.gcd(sr_in, sr_out)
    up = sr_out // gcd
    down = sr_in // gcd
    audio_rs = sps.resample_poly(audio, up, down).astype(np.float32)
    return audio_rs, sr_out

def peak_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = np.max(np.abs(audio)) + eps
    return (audio / m).astype(np.float32)

def float_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16.tobytes()

def vad_trim(audio: np.ndarray, sr: int = 16000, aggressiveness: int = 2,
             frame_ms: int = 30, padding_ms: int = 150) -> np.ndarray:
    """
    Simple VAD trimming using webrtcvad.
    Keeps speech frames and removes long silences.
    """
    if webrtcvad is None:
        return audio

    assert sr in (8000, 16000, 32000, 48000), "webrtcvad supports 8/16/32/48k"
    vad = webrtcvad.Vad(aggressiveness)

    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0:
        return audio

    pcm = float_to_pcm16(audio)

    # Frame slicing in bytes (16-bit)
    bytes_per_sample = 2
    frame_bytes = frame_len * bytes_per_sample

    speech_flags = []
    frames = []
    for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        fb = pcm[i:i + frame_bytes]
        is_speech = vad.is_speech(fb, sr)
        speech_flags.append(is_speech)
        frames.append(fb)

    if not frames:
        return audio

    # Add padding (keep small silences around speech)
    pad_frames = int(padding_ms / frame_ms)
    keep = np.zeros(len(speech_flags), dtype=bool)
    for i, s in enumerate(speech_flags):
        if s:
            start = max(0, i - pad_frames)
            end = min(len(keep), i + pad_frames + 1)
            keep[start:end] = True

    kept_bytes = b"".join([f for f, k in zip(frames, keep) if k])
    if len(kept_bytes) == 0:
        return audio

    kept = np.frombuffer(kept_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    return kept

def preprocess_wav(path: str, target_sr: int = 16000, use_vad: bool = True) -> tuple[np.ndarray, int]:
    audio, sr = read_wav(path)
    audio, sr = resample(audio, sr, target_sr)
    audio = peak_normalize(audio)
    if use_vad:
        audio = vad_trim(audio, sr=sr)
        audio = peak_normalize(audio)
    return audio, sr
