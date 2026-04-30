# Pronunciation Scoring Pipeline

A two-stage pipeline for automatic pronunciation scoring using HuBERT.

## Pipeline Overview

```
Stage 1 — CTC Pretext Task
  HuBERT-Large (LibriSpeech pretrained)
       ↓  fine-tune with CTC on SpeechOcean762 transcripts
  hubert-large-speechocean-ctc/

Stage 2 — Transfer to Scoring Heads  (planned)
  Frozen HuBERT encoder
       ↓
  Phoneme-level score head
  Word-level score head
  Utterance-level score head
```

## Repository Structure

```
pronunciation-scoring/
├── train_hubert_speechocean.py   # Entry point — runs Stage 1 training
├── ctc_training/
│   ├── __init__.py
│   ├── config.py      # Model name, dataset, paths, training hyperparameters
│   ├── processor.py   # Text normalisation, vocab building, Wav2Vec2Processor
│   ├── data.py        # Dataset loading (SpeechOcean762) and preprocessing
│   ├── collator.py    # DataCollatorCTCWithPadding
│   ├── metrics.py     # WER metric factory for HuggingFace Trainer
│   └── model.py       # HuBERT-CTC model builder (frozen CNN encoder)
├── requirements.txt
└── audio/             # Sample audio files for inference / testing
```

## Stage 1 — HuBERT CTC Fine-tuning

**Goal:** Fine-tune HuBERT-Large on SpeechOcean762 transcripts using CTC loss.
The learned representations are then transferred to pronunciation scoring heads.

**Dataset:** [mispeech/speechocean762](https://huggingface.co/datasets/mispeech/speechocean762)
— 5,000 English utterances with human pronunciation scores at phoneme, word, and utterance level.

**Base model:** `facebook/hubert-large-ls960-ft` (HuBERT-Large, LibriSpeech 960h)

**Key design choices:**
- CNN feature encoder is frozen; only transformer layers are fine-tuned.
- Effective batch size 16 via gradient accumulation (4 × 4).
- Gradient checkpointing enabled to reduce VRAM usage (~20% speed cost).
- Best checkpoint selected by lowest validation WER.

### Setup

```bash
pip install transformers datasets jiwer soundfile librosa accelerate
# or
pip install -r requirements.txt
```

### Run training

```bash
python train_hubert_speechocean.py
```

Outputs are saved to `./hubert-large-speechocean-ctc/`:
- model weights and processor
- `test_results.json` with final WER on the held-out test set

### Key hyperparameters (`ctc_training/config.py`)

| Parameter | Value | Notes |
|---|---|---|
| `num_train_epochs` | 30 | |
| `learning_rate` | 1e-4 | linear schedule with 10% warmup |
| `per_device_train_batch_size` | 4 | × 4 gradient accumulation = 16 effective |
| `fp16` | auto | enabled when CUDA is available |
| `MAX_DURATION_SEC` | 20 s | utterances longer than this are dropped |

---

## TODO / Planned Work

- SSL masking strategies for phoneme prediction:
  - Replace phonemes with a mask token
  - Zero 1-dimensional features
  - Zero multi-dimensional prosodic features
- Transfer learning heads for phoneme, word, and utterance-level scores
- Vowel/consonant classification auxiliary task
- Articulation trait prediction
