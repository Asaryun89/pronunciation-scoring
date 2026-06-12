# Pronunciation Scoring Pipeline

A two-stage pipeline for automatic pronunciation scoring using HuBERT.

## Pipeline Overview

```
Stage 1 — CTC Pretext Task                     (stage1_ctc/)
  HuBERT-Large (LibriSpeech pretrained)
       ↓  fine-tune with CTC on SpeechOcean762 transcripts
  outputs/hubert-large-speechocean-ctc/

Stage 2 — Multitask Scoring Model              (stage2_scoring/)
  HuBERT-Large encoder (stage-1 weights or stock hubert-large-ll60k)
       ↓  layer-weighted sum + projection
  Cross-attention fusion with reference-text embedding (Qwen3-Embedding)
       ↓  Transformer refinement → mean pool → MLP head
  Utterance scores: total, accuracy, fluency, prosodic, completeness
```

## Repository Structure

```
pronunciation-scoring/
├── stage1_ctc/            # Stage 1 — HuBERT CTC fine-tuning
│   ├── train.py           # Entry point: python -m stage1_ctc.train
│   ├── config.py          # Model name, dataset, paths, training hyperparameters
│   ├── processor.py       # Text normalisation, vocab building, Wav2Vec2Processor
│   ├── data.py            # Dataset loading (SpeechOcean762) and preprocessing
│   ├── collator.py        # DataCollatorCTCWithPadding
│   ├── metrics.py         # WER metric factory for HuggingFace Trainer
│   └── model.py           # HuBERT-CTC model builder (frozen CNN encoder)
├── stage2_scoring/        # Stage 2 — utterance-level scoring model
│   ├── train.py           # Entry point: python -m stage2_scoring.train
│   ├── hubert_multitask.py# HubertMultiTask model (training + inference forward)
│   ├── scoring_heads.py   # CrossAttentionFusion, MLPScoringHead
│   ├── constants.py       # Score dims/scales shared with inference
│   ├── collate.py         # Batch collator (audio decode, VAD, prosody targets)
│   └── validate.py        # SpeechOcean762 schema validation
├── utils/                 # Shared audio helpers (preprocessing, prosody, alignment)
├── notebooks/
│   └── architecture.ipynb # Visual walkthrough of both stages (diagrams + live introspection)
├── outputs/               # All generated artifacts (gitignored)
│   ├── vocab.json                       # Stage-1 character vocabulary
│   ├── hubert-large-speechocean-ctc/    # Stage-1 model + checkpoints
│   └── ckpt_hubert_multitask/           # Stage-2 checkpoints + result.csv
├── transfer.sh            # rsync helper: sync code/checkpoints with a GPU server
└── requirements.txt
```

**Dataset:** [mispeech/speechocean762](https://huggingface.co/datasets/mispeech/speechocean762)
— 5,000 English utterances with human pronunciation scores at phoneme, word, and utterance level.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install jiwer librosa   # stage 1 extras (WER metric, audio)
```

## Stage 1 — HuBERT CTC Fine-tuning

**Goal:** Fine-tune HuBERT-Large on SpeechOcean762 transcripts using CTC loss.
The learned representations are then transferred to pronunciation scoring heads.

**Base model:** `facebook/hubert-large-ls960-ft` (HuBERT-Large, LibriSpeech 960h)

**Key design choices:**
- CNN feature encoder is frozen; only transformer layers are fine-tuned.
- Gradient checkpointing enabled to reduce VRAM usage (~20% speed cost).
- Best checkpoint selected by lowest validation WER.

```bash
python -m stage1_ctc.train
```

Outputs are saved to `outputs/hubert-large-speechocean-ctc/`:
- model weights and processor
- `test_results.json` with final WER on the held-out test set

All hyperparameters live in `stage1_ctc/config.py`.

## Stage 2 — Utterance Scoring

**Goal:** Train `HubertMultiTask` to predict the five SpeechOcean762
utterance scores (completeness is excluded from the loss — see
`stage2_scoring/constants.py`).

```bash
# Backbone from stage 1 (CTC fine-tuned HuBERT-Large):
python -m stage2_scoring.train --backbone stage1

# Or the stock pretrained HuBERT-Large:
python -m stage2_scoring.train --backbone hubert-large

# Any other checkpoint:
python -m stage2_scoring.train --model <hub-id-or-path>
```

Checkpoints (`best.pt`, selected by Pearson correlation on `total`) and
`result.csv` metrics are written to `outputs/ckpt_hubert_multitask/`.
See `python -m stage2_scoring.train --help` for all options
(text stream, layer unfreezing, prosody auxiliary loss, ...).

## Syncing with a GPU server

```bash
./transfer.sh push-code          # upload source code
./transfer.sh pull-checkpoints   # download outputs/ (models, results)
```

## TODO / Planned Work

- SSL masking strategies for phoneme prediction:
  - Replace phonemes with a mask token
  - Zero 1-dimensional features
  - Zero multi-dimensional prosodic features
- Word- and phoneme-level score heads
- Vowel/consonant classification auxiliary task
- Articulation trait prediction
