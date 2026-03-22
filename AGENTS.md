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
- `sentence_head`: outputs `[0,100]` for `completeness`, `fluency`, `prosodic`, and `total`

### 8) Final Score Fusion (`utils/fusion.py`)

Fused score in `[0,100]`:

`final = alpha * (mean_word*100) + beta * utt_score + gamma * (prosody*100)`

Default weights:

- `alpha=0.6`
- `beta=0.25`
- `gamma=0.15`

## Inference Flow

Implemented in `inference/predictor.py` using the unified `HubertMultiTask` model:

1. Preprocess wav (resample → VAD → normalize)
2. Wav2Vec2FeatureExtractor → `input_values` tensor
3. Faster-Whisper → word timestamps
4. `HubertMultiTask.forward_inference()`:
   - HuBERT hidden states `(1, T, 768)`
   - Estimate `frame_hz = T / duration`
   - ASR-aligned frame spans via `build_word_segments()`
   - Sentence: mean pool → `sentence_head` → sigmoid → `[0,1]`
   - Words: per-span mean pool → `word_head` → sigmoid → `[0,1]`
5. Scale all scores `× 10` → `[0, 10]` (SpeechOcean native scale)
6. Extract prosody features (audio-computed, stored in metadata)
7. Return dataset-shaped structured JSON

Output keys:

- `accuracy`, `completeness`, `fluency`, `prosodic`, `total` — utterance scores `[0, 10]`
- `text` — ASR transcript
- `words` — per-word: `text`, `accuracy`, `total`, `start`, `end`, `asr_prob`
- `audio` — `path`
- `inference_metadata` — `language`, `duration`, `frame_hz`, `prosody_features`, `scoring_validity`

## Scoring Validity

`inference_metadata.scoring_validity` reflects the checkpoint state:

- `"untrained_random_init"` — no checkpoint loaded; scores are meaningless
- `"trained:<path>"` — loaded from a trained checkpoint; scores are meaningful

Load a checkpoint via `PredictorConfig(checkpoint_path="ckpt_hubert_multitask/best.pt")`.

## Unified Model Architecture (`models/train.py` → `HubertMultiTask`)

All four heads predict `[0, 1]` via sigmoid. Multiply by scale to display:

| Head | Output | Scale | Trained on |
|---|---|---|---|
| `sentence_head` | 5 scalars | ×10 | `total, accuracy, fluency, prosodic, completeness` |
| `prosody_feat_head` | 5 scalars | raw | `rms, zcr, peak_rate, f0_mean, f0_std` |
| `word_head` | 1 per word | ×10 | `words[].total` |
| `phone_head` | 1 per phone | ×2 | `words[].phones-accuracy` |

Score dimension order constant: `SENT_DIMS = ["total", "accuracy", "fluency", "prosodic", "completeness"]`

Word segmentation: even frame splits during training; ASR-aligned spans at inference.

## Repository Layout

- `models/train.py` — `HubertMultiTask` (unified train+inference model), `Collator`, `compute_loss`, `eval_epoch`, `main()`
- `models/asr_aligner.py` — Faster-Whisper word timestamps
- `models/hubert_encoder.py` — standalone HuBERT encoder (retained for compatibility)
- `models/scoring_heads.py` — `MultiHeadScorer` (superseded by `HubertMultiTask`; retained for reference)
- `utils/` — preprocessing, alignment, pooling, prosody, fusion
- `inference/` — `predictor.py`, `infer.py` (CLI), `api.py` (FastAPI)
- `audio/` — sample wav files
- `config/` — currently empty

## Run Commands

Training on SpeechOcean762:

```bash
python models/train.py \
  --dataset mispeech/speechocean762 \
  --epochs 10 \
  --batch_size 4 \
  --out_dir ckpt_hubert_multitask
```

CLI inference (with trained checkpoint):

```bash
python -m inference.infer \
  --wav "audio/learner/01_learner.wav" \
  --lang en \
  --checkpoint ckpt_hubert_multitask/best.pt
```

CLI inference (random weights, for pipeline testing):

```bash
python -m inference.infer --wav "audio/learner/01_learner.wav" --lang en
```

FastAPI server:

```bash
CHECKPOINT_PATH=ckpt_hubert_multitask/best.pt \
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

## Dataset Inspection Notes

The notebook [data/data_inspection.ipynb](/mnt/d/pronunciation_scoring/data/data_inspection.ipynb) is the current inspection entrypoint for the Hugging Face dataset used in training preparation:

- Loads `mispeech/speechocean762` with `load_dataset(...)`
- Casts the `audio` column to `Audio(decode=False)` before row inspection
- Prints split sizes, feature schema, sample keys, utterance-level fields, first word annotation, and audio metadata

This `decode=False` step is intentional:

- it avoids `torchcodec` / FFmpeg runtime failures during dataset inspection
- it preserves file/path metadata needed for schema review before training collation

For training preparation, treat the inspected dataset annotation as:

- utterance labels: `accuracy`, `completeness`, `fluency`, `prosodic`, `total`
- utterance metadata: `text`, `speaker`, `gender`, `age`
- word annotations: `text`, `accuracy`, `stress`, `total`, `phones`, `phones-accuracy`, `mispronunciations`
- audio metadata payload: at minimum `path`, and optionally `bytes` or decoded array data depending on loader settings

The validation and collation logic in `models/train.py` should stay aligned with this inspected schema.

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
