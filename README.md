# Pronunciation Scoring — Inference Package

An end-to-end pronunciation scoring pipeline built on HuBERT + Faster-Whisper.
Accepts a WAV file (or raw numpy array) and returns utterance-level and word-level pronunciation scores.

---

## Architecture

```
WAV → Preprocess → HuBERT encoder ─┐
                                    ├─→ HubertMultiTask → scores (0–10)
           Whisper ASR ─────────────┘
           (word timestamps + transcript)
```

The model (`HubertMultiTask`) was fine-tuned on SpeechOcean762 and outputs five utterance scores on the 0–10 scale:
`accuracy`, `completeness`, `fluency`, `prosodic`, `total`.

---

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd pronunciation-scoring
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU (recommended):** Replace the CPU torch wheel with a CUDA-enabled one:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu128
> ```

### 3. Download the checkpoint

Place the trained model weights at:
```
ckpt_hubert_multitask/best.pt
```

The checkpoint is not distributed in this repo (binary too large). Obtain it from the team's shared storage or contact the maintainer.

---

## Quick start

### CLI

```bash
python -m inference.infer --wav path/to/audio.wav --lang en
```

With an explicit checkpoint path:

```bash
python -m inference.infer --wav path/to/audio.wav --lang en \
    --checkpoint ckpt_hubert_multitask/best.pt
```

### Python API

```python
from inference.predictor import PronunciationPredictor, PredictorConfig

predictor = PronunciationPredictor(PredictorConfig(
    checkpoint_path = "ckpt_hubert_multitask/best.pt",
    device          = "cuda",   # or "cpu"
    whisper_size    = "small",
    language        = "en",
))

result = predictor.predict("path/to/audio.wav", language="en")
print(result["total"])       # overall score 0–10
print(result["accuracy"])    # accuracy sub-score 0–10
```

### Notebook

Open [run_inference.ipynb](run_inference.ipynb) for an interactive demo with waveform visualization and a word-by-word score table.

---

## FastAPI server

```bash
CHECKPOINT_PATH=ckpt_hubert_multitask/best.pt uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

**POST `/score`**

| Field      | Type   | Description                     |
|------------|--------|---------------------------------|
| `file`     | file   | WAV audio file (multipart form) |
| `language` | string | ISO-639-1 code, default `"en"`  |

---

## Output schema

```json
{
  "accuracy":     8.2,
  "completeness": 9.0,
  "fluency":      7.5,
  "prosodic":     7.8,
  "total":        8.1,
  "text":         "she sells sea shells",
  "words": [
    {
      "text":    "she",
      "start":   0.12,
      "end":     0.40,
      "asr_prob": 0.97,
      "accuracy": null,
      "total":    null
    }
  ],
  "audio": { "path": "path/to/audio.wav" },
  "inference_metadata": {
    "language":         "en",
    "duration":         2.4,
    "frame_hz":         50,
    "prosody_features": { "f0_mean": 180.3, "speaking_rate": 4.1, "...": "..." },
    "scoring_validity": "trained:ckpt_hubert_multitask/best.pt"
  }
}
```

> **Note:** `words[].accuracy` and `words[].total` are `null` — word-level scoring requires a separate word-level model head not yet trained. Word entries carry timestamps (`start`, `end`) and ASR confidence (`asr_prob`) only.

---

## Repository layout

```
inference/
  api.py          — FastAPI app (POST /score)
  predictor.py    — PronunciationPredictor class (main entry point)
  infer.py        — CLI wrapper

models/
  hubert_multitask.py  — HubertMultiTask model definition
  asr_aligner.py       — Faster-Whisper wrapper
  scoring_heads.py     — utterance scoring MLP heads
  constants.py         — shared dimension/scale constants
  ctc_aligner.py       — CTC forced aligner (optional, not used by default)

utils/
  preprocessing.py — resample, VAD, normalize
  prosody.py       — F0 / energy / rate feature extraction
  alignment.py     — frame-to-word span mapping

notebook_infer/
  pipeline.py      — score_file() / score_array() helpers for notebooks
  display.py       — rich display utilities (waveform plot, score table)

run_inference.ipynb  — interactive demo notebook
```

---

## Requirements

- Python 3.10+
- PyTorch 2.x (CPU or CUDA)
- faster-whisper, transformers, soundfile, webrtcvad
- fastapi + uvicorn (for the API server)

See [requirements.txt](requirements.txt) for the full pinned list.
