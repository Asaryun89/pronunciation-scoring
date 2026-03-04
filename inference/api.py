from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil
import json

from inference.predictor import PronunciationPredictor, PredictorConfig

app = FastAPI(title="Pronunciation Scoring API (HuBERT + ASR Alignment)")

predictor = PronunciationPredictor(PredictorConfig(device="cpu", whisper_device="cpu"))

@app.post("/score")
async def score(file: UploadFile = File(...), language: str = "en"):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with tmp as f:
            shutil.copyfileobj(file.file, f)
        path = tmp.name

    result = predictor.predict(path, language=language)
    return result