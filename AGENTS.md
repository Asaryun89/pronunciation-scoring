# AGENTS.md — Integration Guide for Web Builders

This document is for teammates integrating the pronunciation scoring engine into a web product (frontend, backend, or full-stack).

---

## What this package does

Given a WAV audio clip, it returns:

- **5 utterance-level scores** (0–10): `accuracy`, `completeness`, `fluency`, `prosodic`, `total`
- **Transcript** from Whisper ASR
- **Per-word timestamps** (`start`, `end` in seconds) and ASR confidence (`asr_prob`)
- **Prosody metadata**: F0 mean/std, speaking rate, energy, pause ratio

Word-level accuracy scores (`words[].accuracy`, `words[].total`) are `null` in the current model — the utterance-level scores are the reliable outputs to display to users.

---

## Option A — Use the FastAPI server (recommended for web)

The fastest integration path. Your frontend or backend calls a single HTTP endpoint.

### Start the server

```bash
# Install deps once
pip install -r requirements.txt

# Run (set checkpoint path)
CHECKPOINT_PATH=ckpt_hubert_multitask/best.pt uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

Windows PowerShell:
```powershell
$env:CHECKPOINT_PATH="ckpt_hubert_multitask/best.pt"
uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

### Call from your web backend

```python
import httpx

with open("user_recording.wav", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/score",
        files={"file": ("audio.wav", f, "audio/wav")},
        params={"language": "en"},
    )
result = response.json()
print(result["total"])     # 0–10
print(result["text"])      # transcript
```

### Call from JavaScript / fetch

```js
const formData = new FormData();
formData.append("file", audioBlob, "audio.wav");

const res = await fetch("http://localhost:8000/score?language=en", {
  method: "POST",
  body: formData,
});
const result = await res.json();
console.log(result.total, result.text);
```

### Interactive API docs

Visit `http://localhost:8000/docs` to test the endpoint in the browser.

---

## Option B — Import `PronunciationPredictor` directly (Python backend)

Use this when your backend is already Python (Django, Flask, FastAPI, etc.) and you want to embed the scorer in-process.

```python
from inference.predictor import PronunciationPredictor, PredictorConfig

# Create once at startup — loading is slow (~10–20 s)
predictor = PronunciationPredictor(PredictorConfig(
    checkpoint_path = "ckpt_hubert_multitask/best.pt",
    device          = "cuda",       # "cpu" if no GPU
    whisper_size    = "small",      # "tiny" for speed, "medium" for accuracy
    whisper_device  = "cuda",
    language        = "en",
))

# Call per request — fast (~0.5–2 s on GPU)
result = predictor.predict("path/to/uploaded.wav", language="en")
```

**Important:** Instantiate `PronunciationPredictor` once (e.g. at app startup or as a module-level singleton). Each instantiation loads HuBERT and Whisper from disk.

---

## Output schema reference

```json
{
  "accuracy":     8.2,
  "completeness": 9.0,
  "fluency":      7.5,
  "prosodic":     7.8,
  "total":        8.1,

  "text": "she sells sea shells",

  "words": [
    {
      "text":              "she",
      "start":             0.12,
      "end":               0.40,
      "asr_prob":          0.97,
      "accuracy":          null,
      "total":             null,
      "stress":            null,
      "phones":            [],
      "phones-accuracy":   [],
      "mispronunciations": []
    }
  ],

  "audio": {
    "path": "path/to/audio.wav"
  },

  "inference_metadata": {
    "language":    "en",
    "duration":    2.4,
    "frame_hz":    50,
    "prosody_features": {
      "f0_mean":       180.3,
      "f0_std":        40.1,
      "speaking_rate": 4.1,
      "energy_mean":   0.05,
      "pause_ratio":   0.12
    },
    "scoring_validity": "trained:ckpt_hubert_multitask/best.pt"
  }
}
```

### Score interpretation

| Score        | Meaning                                      |
|--------------|----------------------------------------------|
| `total`      | Overall pronunciation quality (0–10)         |
| `accuracy`   | Phonemic correctness                         |
| `completeness` | How completely the sentence was said       |
| `fluency`    | Speaking smoothness and pace                 |
| `prosodic`   | Pitch, rhythm, stress patterns               |

Scores follow the SpeechOcean762 annotation scale (0–10, higher is better).

---

## Audio requirements

| Property    | Required value                        |
|-------------|---------------------------------------|
| Format      | WAV (PCM), MP3, FLAC — anything soundfile reads |
| Sample rate | Any — resampled to 16 kHz internally  |
| Channels    | Mono or stereo — mixed to mono internally |
| Duration    | 1–30 s recommended                    |

The pipeline applies VAD (Voice Activity Detection) to strip silence automatically.

---

## Performance notes

| Setting                 | Latency (RTX 4050) | Latency (CPU only) |
|-------------------------|--------------------|--------------------|
| `whisper_size="tiny"`   | ~0.4 s             | ~2–4 s             |
| `whisper_size="small"`  | ~0.8 s             | ~4–8 s             |
| `whisper_size="medium"` | ~1.5 s             | ~10–20 s           |

- First call after startup is slower (model JIT warm-up).
- For a web API, keep the predictor alive across requests (do not recreate per request).

---

## Environment variables (for the FastAPI server)

| Variable          | Default | Description                           |
|-------------------|---------|---------------------------------------|
| `CHECKPOINT_PATH` | `None`  | Path to `best.pt` — required for real scores |

---

## Key files to know

| File                        | What it does                                          |
|-----------------------------|-------------------------------------------------------|
| `inference/api.py`          | FastAPI app — single POST `/score` endpoint           |
| `inference/predictor.py`    | `PronunciationPredictor` — the core scoring class     |
| `inference/infer.py`        | CLI entry point                                       |
| `models/hubert_multitask.py`| Model architecture (HuBERT + scoring heads)           |
| `notebook_infer/pipeline.py`| `score_file()` / `score_array()` for notebooks/demos |

---

## Common integration patterns

### Django / Flask — singleton predictor

```python
# myapp/scoring.py
from inference.predictor import PronunciationPredictor, PredictorConfig

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = PronunciationPredictor(PredictorConfig(
            checkpoint_path="ckpt_hubert_multitask/best.pt",
            device="cuda",
        ))
    return _predictor
```

```python
# myapp/views.py
from .scoring import get_predictor

def score_view(request):
    audio_file = request.FILES["audio"]
    # save to temp file, then:
    result = get_predictor().predict(tmp_path, language="en")
    return JsonResponse(result)
```

### Scoring from a bytes buffer (no disk write)

```python
import numpy as np, soundfile as sf, io

audio_bytes: bytes = ...  # raw wav bytes from upload
audio, sr = sf.read(io.BytesIO(audio_bytes))
audio = audio.astype(np.float32)
if audio.ndim == 2:
    audio = audio.mean(axis=1)  # stereo → mono

from notebook_infer.pipeline import score_array, ScoreConfig
result = score_array(audio, sr, predictor, ScoreConfig())
```
