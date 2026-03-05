# AGENTS.md

## Project: Pronunciation Scoring (Inference-First)

This repository currently implements an **inference-oriented pronunciation scoring pipeline** using:

- `HuBERT` embeddings (`facebook/hubert-base-ls960`)
- `Faster-Whisper` word timestamp alignment
- lightweight pooling/prosody feature extraction
- a multi-head MLP scorer and score fusion

The primary entrypoints are under `inference/`.

## Current Architecture (As Implemented)

### 1) Audio Preprocessing (`utils/preprocessing.py`)

Pipeline:

1. Read wav with `soundfile`
2. Convert to mono
3. Resample with `scipy.signal.resample_poly` to 16 kHz
4. Peak normalize
5. Optional VAD trim with `webrtcvad`
6. Peak normalize again

Key function:

- `preprocess_wav(path, target_sr=16000, use_vad=True)`

### 2) Speech Representation (`models/hubert_encoder.py`)

- Uses `Wav2Vec2FeatureExtractor` + `HubertModel`
- Default model: `facebook/hubert-base-ls960`
- Returns frame-level embeddings `(T, D)` and estimated `frame_hz`

Key classes:

- `HubertConfig`
- `HubertEncoder.encode(audio, sr)`

### 3) Timestamp Alignment (`models/asr_aligner.py`)

- Uses `faster-whisper` (`WhisperModel`) for transcript + word timestamps
- `word_timestamps=True`
- Returns dictionary with:
  - `text`
  - `words` (word/start/end/prob)
  - `language`
  - `duration`

Key classes:

- `ASRConfig`
- `ASRAligner.transcribe_with_timestamps(...)`

### 4) Frame-to-Word Mapping (`utils/alignment.py`)

- Converts timestamp seconds to frame indices using `frame_hz`
- Builds valid `[i0, i1)` frame spans per word

Key function:

- `build_word_segments(word_ts, frame_hz, T)`

### 5) Pooling (`utils/pooling.py`)

- `mean_pool(emb, i0, i1)` for segment vectors
- `utt_pool_mean_std(emb)` for utterance vector (`2D` dimension)

### 6) Prosody Features (`utils/prosody.py`)

Extracted features:

- `rms`
- `zcr`
- `peak_rate`
- `f0_mean` and `f0_std` (if `pyworld` available, else 0)

Key functions:

- `basic_prosody_features(audio, sr)`
- `prosody_to_vector(feats)`

### 7) Scoring Heads (`models/scoring_heads.py`)

`MultiHeadScorer` contains:

- `word_head`: outputs `[0,1]` via sigmoid
- `utt_head`: outputs `[0,100]` via sigmoid * 100
- `prosody_head`: outputs `[0,1]` via sigmoid

### 8) Final Score Fusion (`utils/fusion.py`)

Fused score in `[0,100]`:

`final = alpha * (mean_word*100) + beta * utt_score + gamma * (prosody*100)`

Default weights:

- `alpha=0.6`
- `beta=0.25`
- `gamma=0.15`

## Inference Flow

Implemented in `inference/predictor.py`:

1. Preprocess wav
2. Encode HuBERT embeddings
3. ASR word timestamps
4. Map words to HuBERT frame spans
5. Pool word embeddings + utterance embedding
6. Extract prosody vector
7. Run multi-head scorer
8. Fuse scores
9. Return structured JSON

Output keys:

- `text`
- `overall_score`
- `accuracy`
- `fluency`
- `prosody`
- `prosody_features`
- `words` (with per-word score)

## Important Current Limitation

The scorer in `PronunciationPredictor` is **randomly initialized** on first use and set to eval mode.

- There is currently **no checkpoint loading** in `inference/predictor.py`
- Scores are not meaningful for production until trained weights are added

Agents must not present current predictions as validated assessment scores.

## Repository Layout

- `models/`: HuBERT encoder, ASR aligner, scoring heads
- `utils/`: preprocessing, alignment, pooling, prosody, fusion
- `inference/`: predictor, CLI infer script, FastAPI app, local test script
- `audio/`: sample native/learner wav files
- `config/`: currently empty

## Run Commands

CLI inference:

```bash
python -m inference.infer --wav "audio/learner/01_learner.wav" --lang en
```

FastAPI server:

```bash
uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

Example request behavior:

- `POST /score` with uploaded wav
- optional query param: `language=en`

## Dependencies

From `requirements.txt`:

- `numpy`, `scipy`
- `torch`, `torchaudio`, `transformers`
- `soundfile`, `webrtcvad`
- `faster-whisper`
- `fastapi`, `uvicorn`, `pydantic`

`pyworld` is optional in code (not pinned in requirements).


## Data Representation (Training Sample)

Use the following structure as the canonical sample-level representation for dataset rows:

- Utterance-level fields:
  - `accuracy` (int)
  - `completeness` (float)
  - `fluency` (int)
  - `prosodic` (int)
  - `total` (int)
  - `text` (str)
  - `speaker` (str)
  - `gender` (str)
  - `age` (int)
- `words` (list[dict]), each item includes:
  - `text` (str)
  - `accuracy` (int)
  - `stress` (int)
  - `total` (int)
  - `phones` (list[str])
  - `phones-accuracy` (list[float])
  - `mispronunciations` (list)
- `audio` (dict):
  - `path` (str)
  - `bytes` (bytes)

Example shape:

```python
{
    'accuracy': 8,
    'completeness': 10.0,
    'fluency': 9,
    'prosodic': 9,
    'text': 'WE CALL IT BEAR',
    'total': 8,
    'words': [
        {
            'accuracy': 10,
            'phones': ['W', 'IY0'],
            'phones-accuracy': [2.0, 2.0],
            'stress': 10,
            'text': 'WE',
            'total': 10,
            'mispronunciations': []
        }
    ],
    'speaker': '0001',
    'gender': 'm',
    'age': 6,
    'audio': {'bytes': b'...', 'path': '000010011.wav'}
}
```

## Agent Guidelines For This Repo

1. Keep changes modular
- Put model components in `models/`
- Put signal/feature helpers in `utils/`
- Keep orchestration in `inference/`

2. Preserve I/O contracts
- `predict()` should continue returning the current output schema unless explicitly changed
- If schema changes, update CLI/API and docs together

3. Be explicit about scoring validity
- If untrained weights are used, document this in code comments and README/AGENTS updates
- If adding checkpoint support, validate load path and device mapping

4. Prefer CPU-safe defaults
- Current defaults are CPU-friendly (`device="cpu"`, Whisper `int8`)
- Keep defaults stable unless the user requests GPU-first behavior

5. Add minimal tests when changing behavior
- Validate preprocessing shape/range assumptions
- Validate mapping from seconds to frame indices
- Validate fused score clamping `[0,100]`

6. Avoid speculative docs
- Document only what exists in this repository
- Mark proposed features as future work

## Suggested Next Extension

If you implement training next, add:

- dataset definition and label format
- training loop and loss definitions
- checkpoint save/load
- evaluator metrics (MAE/RMSE/Pearson/Spearman)
- config files under `config/`
