# AGENTS.md — Reproducibility Guide

This document tells AI agents (and humans) everything needed to understand,
run, and extend this repository without additional context.

---

## Project Purpose

Automatic pronunciation scoring using HuBERT, in two stages:

- **Stage 1** (`stage1_ctc/`) fine-tunes HuBERT-Large with CTC on
  SpeechOcean762 transcripts (text-prediction pretext task).
- **Stage 2** (`stage2_scoring/`) trains `HubertMultiTask` — the (optionally
  stage-1-initialised) HuBERT encoder fused with a reference-text embedding —
  to predict utterance-level pronunciation scores.

---

## Environment

- Python 3.9+
- CUDA GPU strongly recommended (tested on CUDA 11.8+)
- Install deps: `pip install -r requirements.txt` plus `jiwer librosa` for Stage 1

---

## Repository Layout

| Path | Purpose |
|---|---|
| `stage1_ctc/train.py` | Stage 1 entry point (`python -m stage1_ctc.train`) |
| `stage1_ctc/config.py` | All Stage 1 hyperparameters and path constants |
| `stage1_ctc/processor.py` | Text normalisation, vocab file creation, `Wav2Vec2Processor` |
| `stage1_ctc/data.py` | Dataset download, train/val split, per-sample preprocessing |
| `stage1_ctc/collator.py` | Batching with independent CTC padding for inputs and labels |
| `stage1_ctc/metrics.py` | WER metric factory passed to `Trainer` |
| `stage1_ctc/model.py` | `HubertForCTC` loader with frozen CNN encoder |
| `stage2_scoring/train.py` | Stage 2 entry point (`python -m stage2_scoring.train`) |
| `stage2_scoring/hubert_multitask.py` | `HubertMultiTask` model (training + inference forward) |
| `stage2_scoring/scoring_heads.py` | `CrossAttentionFusion`, `MLPScoringHead` building blocks |
| `stage2_scoring/constants.py` | Score dims/scales shared between training and inference |
| `stage2_scoring/collate.py` | Batch collator: audio decode → VAD trim → prosody targets |
| `stage2_scoring/validate.py` | SpeechOcean762 schema validation (run on a sample pre-training) |
| `utils/` | Shared audio helpers: preprocessing, prosody features, alignment |
| `notebooks/architecture.ipynb` | Visual architecture walkthrough of both stages (diagrams + live model introspection) |
| `outputs/` | All generated artifacts: vocab, checkpoints, results (gitignored) |
| `transfer.sh` | rsync helper to sync code/checkpoints with a remote GPU server |

---

## How to Reproduce Stage 1 Training

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt jiwer librosa

# 3. Run training (from the repository root)
python -m stage1_ctc.train
```

The script will:
1. Download `mispeech/speechocean762` from HuggingFace Hub (~1 GB).
2. Build `outputs/vocab.json` from training transcripts (skipped if already present).
3. Preprocess and cache the dataset (multiprocessing, 4 workers).
4. Fine-tune `facebook/hubert-large-ls960-ft` for 30 epochs.
5. Save the best checkpoint (by val WER) to `outputs/hubert-large-speechocean-ctc/`.
6. Write `test_results.json` with the final test-set WER.

---

## How to Reproduce Stage 2 Training

```bash
# Backbone = stage-1 CTC fine-tuned encoder (default dir: outputs/hubert-large-speechocean-ctc)
python -m stage2_scoring.train --backbone stage1

# Backbone = stock pretrained HuBERT-Large (facebook/hubert-large-ll60k)
python -m stage2_scoring.train --backbone hubert-large

# Any explicit checkpoint (overrides --backbone)
python -m stage2_scoring.train --model <hub-id-or-path>
```

Notes:
- Both backbone options are HuBERT-Large (24 layers, hidden 1024).
  `--num_unfreeze_hubert_layers` defaults to 12 = top half; pass 24 for full fine-tuning.
- The text stream uses `Qwen/Qwen3-Embedding-0.6B` (frozen) by default; disable with `--text_model ''`.
- Completeness (index 4) is excluded from the loss (`ACTIVE_SENT_IDXS` in
  `stage2_scoring/constants.py`) because SpeechOcean learners almost always score 10/10.
- Best checkpoint (`best.pt`, by `pearson_total`) and `result.csv` go to
  `outputs/ckpt_hubert_multitask/`.

---

## Module Responsibilities (Stage 1)

### `stage1_ctc/config.py`
Single source of truth for all Stage 1 constants. Edit here to change model,
dataset, output directory, or any training hyperparameter.

### `stage1_ctc/processor.py`
- `normalise_text(text)` — lowercase, strip punctuation except apostrophe,
  collapse whitespace.
- `build_vocab(dataset, vocab_path)` — iterates training transcripts, builds a
  character-level vocab at `outputs/vocab.json` with `|` (word boundary), `[UNK]`, `[PAD]`.
- `get_processor(vocab_path)` — constructs `Wav2Vec2Processor` from the vocab
  file. Must be called after `build_vocab`.

### `stage1_ctc/data.py`
- `load_speechocean()` — downloads dataset, resamples audio to 16 kHz,
  carves 10% validation split from train.
- `make_preprocess_fn(processor)` — returns a closure that converts a raw
  sample to `{input_values, attention_mask, labels}`. Samples longer than
  `MAX_DURATION_SEC` return `None` fields and are later filtered out.
- `prepare_dataset(dataset, processor)` — applies preprocessing via
  `dataset.map` and removes too-long rows.

### `stage1_ctc/collator.py`
`DataCollatorCTCWithPadding` — pads `input_values` and `labels` separately
(required by CTC). Label padding uses `-100` so loss ignores pad positions.

### `stage1_ctc/metrics.py`
`make_compute_metrics(processor)` — returns a function that decodes predicted
and reference token IDs and computes WER via `jiwer`.

### `stage1_ctc/model.py`
`build_model(processor)` — loads `HubertForCTC`, re-initialises the LM head
to match the vocabulary size, and freezes the CNN feature encoder.
To enable full fine-tuning, remove `model.freeze_feature_encoder()`.

---

## Module Responsibilities (Stage 2)

### `stage2_scoring/hubert_multitask.py`
`HubertMultiTask` — HuBERT encoder with learnable layer-weighted sum, optional
frozen text stream (mean-pooled reference transcript), cross-attention fusion
(audio Q, text K/V), post-fusion Transformer, MLP head emitting 5 sigmoid
scores, plus an auxiliary prosody regression head. `forward()` is the training
path; `forward_inference()` additionally maps ASR word timestamps to frame spans.

### `stage2_scoring/collate.py`
`Collator` — decodes raw audio bytes, resamples to 16 kHz, peak-normalises,
VAD-trims (webrtcvad), computes prosody targets, zero-mean/unit-std normalises,
pads, and tokenises reference text. Mirrors the inference preprocessing chain.

### `stage2_scoring/constants.py`
`SENT_DIMS`, score scales, `ACTIVE_SENT_IDXS`, `PROSODY_DIMS`,
`HUBERT_FRAME_HZ` — shared between training and inference. Any change here
must be reflected in both pipelines.

---

## Common Issues

| Symptom | Fix |
|---|---|
| CUDA out of memory (Stage 1) | Reduce `per_device_train_batch_size` to 2 or disable `gradient_checkpointing` |
| CUDA out of memory (Stage 2) | Reduce `--batch_size` to 2 or `--num_unfreeze_hubert_layers` |
| `vocab.json` mismatch | Delete `outputs/vocab.json` and rerun — it will be rebuilt |
| Slow preprocessing | Increase `num_proc` in `prepare_dataset` or cache with `dataset.save_to_disk` |
| WER not improving | Try unfreezing CNN encoder (remove `freeze_feature_encoder()`) or lowering LR |
| `--backbone stage1` dir not found | Run Stage 1 first, or point `--stage1_dir` at the checkpoint directory |
