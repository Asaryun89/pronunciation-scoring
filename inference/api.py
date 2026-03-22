import os
from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil

from inference.predictor import PronunciationPredictor, PredictorConfig

app = FastAPI(title="Pronunciation Scoring API (HuBERT + ASR Alignment)")

# Set CHECKPOINT_PATH env var to load a trained model.
# Example: CHECKPOINT_PATH=ckpt_hubert_multitask/best.pt uvicorn inference.api:app ...
_ckpt = os.environ.get("CHECKPOINT_PATH", None)
predictor = PronunciationPredictor(
    PredictorConfig(device="cpu", whisper_device="cpu", checkpoint_path=_ckpt)
)

@app.post("/score")
async def score(file: UploadFile = File(...), language: str = "en"):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with tmp as f:
            shutil.copyfileobj(file.file, f)
        path = tmp.name

    result = predictor.predict(path, language=language)
    return result