"""
EchoScore — FastAPI Backend
Chạy từ thư mục gốc `ai/`:
    uvicorn app:app --reload --port 8000
"""

import os
import tempfile
import shutil
import time

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference.predictor import PronunciationPredictor, PredictorConfig

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="EchoScore API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ────────────────────────────────────────────────────────────────
_ckpt = os.environ.get("CHECKPOINT_PATH", "ckpt_hubert_multitask/best_2.pt")
_text_model = os.environ.get("TEXT_MODEL_NAME") or None
_hubert_name = os.environ.get("HUBERT_MODEL_NAME", "facebook/hubert-base-ls960")

predictor = PronunciationPredictor(
    PredictorConfig(
        device           = "cuda" if torch.cuda.is_available() else "cpu",
        hubert_name      = _hubert_name,
        whisper_size     = "small",
        whisper_device   = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_path  = _ckpt if os.path.exists(_ckpt) else None,
        text_model_name  = _text_model,
    )
)

print(f"✅ Model ready  |  checkpoint: {_ckpt}  |  device: {predictor.cfg.device}")


def safe_remove(path: str, retries: int = 5, delay: float = 0.3):
    """Xóa file — trên Windows đôi khi cần đợi process release handle."""
    for i in range(retries):
        try:
            os.remove(path)
            return
        except PermissionError:
            if i < retries - 1:
                time.sleep(delay)
    # Nếu vẫn không xóa được thì bỏ qua, không crash server
    pass


# ── Endpoint /score ───────────────────────────────────────────────────────────
@app.post("/score")
async def score_audio(file: UploadFile = File(...), language: str = "en"):
    raw_path = None
    wav_path = None

    try:
        # Bước 1: lưu file gốc (webm/mp3/wav...)
        suffix = os.path.splitext(file.filename or "audio.wav")[-1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            raw_path = tmp.name

        # Bước 2: convert sang WAV 16kHz mono
        wav_path = raw_path + "_converted.wav"
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(raw_path)
            seg = seg.set_channels(1).set_frame_rate(16000)
            seg.export(wav_path, format="wav")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Không decode được audio: {e}"
            )
        finally:
            # Xóa file gốc ngay sau khi đã convert xong
            if raw_path and os.path.exists(raw_path):
                safe_remove(raw_path)
                raw_path = None

        # Bước 3: chạy model — KHÔNG dùng finally để xóa wav_path ở đây
        # vì model có thể vẫn giữ file handle sau khi return
        result = predictor.predict(wav_path, language=language)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        # Dọn file gốc nếu chưa xóa
        if raw_path and os.path.exists(raw_path):
            safe_remove(raw_path)

    # Xóa WAV sau khi result đã được build xong (model đã release handle)
    if wav_path and os.path.exists(wav_path):
        safe_remove(wav_path)

    return result


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":     "ok",
        "device":     predictor.cfg.device,
        "checkpoint": predictor._scoring_validity,
    }
