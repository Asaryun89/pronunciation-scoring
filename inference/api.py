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
    PredictorConfig(
        device="cpu",
        hubert_name=os.environ.get("HUBERT_MODEL_NAME", "facebook/hubert-base-ls960"),
        whisper_device="cpu",
        checkpoint_path=_ckpt,
        text_model_name=os.environ.get("TEXT_MODEL_NAME") or None,
    )
)

@app.post("/score")
async def score(file: UploadFile = File(...), language: str = "en"):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    try:
        result = predictor.predict(path, language=language)
    finally:
        os.remove(path)
    return result
