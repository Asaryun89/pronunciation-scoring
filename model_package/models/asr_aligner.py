from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from faster_whisper import WhisperModel

@dataclass
class ASRConfig:
    model_size: str = "small"   # or "base", "tiny"
    device: str = "cpu"         # "cuda" if available
    compute_type: str = "int8"  # "float16" on GPU

class ASRAligner:
    """
    Uses Faster-Whisper to get transcript + word timestamps.
    Practical alignment: word-level segments (start/end seconds).
    """
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self.model = WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)

    def transcribe_with_timestamps(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = "en",
        beam_size: int = 5
    ) -> Dict[str, Any]:
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=False  # we already do VAD in preprocessing
        )

        words: List[Dict[str, Any]] = []
        full_text = []

        for seg in segments:
            if seg.text:
                full_text.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    # w.word includes leading space sometimes
                    words.append({
                        "word": w.word.strip(),
                        "start": float(w.start),
                        "end": float(w.end),
                        "prob": float(w.probability) if w.probability is not None else None
                    })

        return {
            "text": " ".join([t for t in full_text if t]),
            "words": words,
            "language": getattr(info, "language", None),
            "duration": getattr(info, "duration", None),
        }