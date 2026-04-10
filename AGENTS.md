# AGENTS.md — Reproducibility Guide

This document tells AI agents (and humans) everything needed to understand,
run, and extend this repository without additional context.

---

## Project Purpose

Automatic pronunciation scoring using HuBERT.
Stage 1 fine-tunes HuBERT-Large with CTC on SpeechOcean762 transcripts.
The resulting encoder is intended for transfer to phoneme/word/utterance scoring heads (Stage 2, not yet implemented).

---

## Environment

- Python 3.9+
- CUDA GPU strongly recommended (tested on CUDA 11.8+)
- Install deps: `pip install transformers datasets jiwer soundfile librosa accelerate`

---

## Repository Layout

| Path | Purpose |
|---|---|
| `train_hubert_speechocean.py` | Main entry point; orchestrates all stages |
| `ctc_training/config.py` | All hyperparameters and path constants |
| `ctc_training/processor.py` | Text normalisation, vocab file creation, `Wav2Vec2Processor` |
| `ctc_training/data.py` | Dataset download, train/val split, per-sample preprocessing |
| `ctc_training/collator.py` | Batching with independent CTC padding for inputs and labels |
| `ctc_training/metrics.py` | WER metric factory passed to `Trainer` |
| `ctc_training/model.py` | `HubertForCTC` loader with frozen CNN encoder |
| `audio/` | Sample audio files for quick inference checks |
| `ckpt_hubert_multitask/` | Checkpoint directory (not tracked by git) |

---

## How to Reproduce Stage 1 Training

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install transformers datasets jiwer soundfile librosa accelerate

# 3. Run training
python train_hubert_speechocean.py
```

The script will:
1. Download `mispeech/speechocean762` from HuggingFace Hub (~1 GB).
2. Build `vocab.json` from training transcripts (skipped if already present).
3. Preprocess and cache the dataset (multiprocessing, 4 workers).
4. Fine-tune `facebook/hubert-large-ls960-ft` for 30 epochs.
5. Save the best checkpoint (by val WER) to `./hubert-large-speechocean-ctc/`.
6. Write `test_results.json` with the final test-set WER.

---

## Module Responsibilities

### `ctc_training/config.py`
Single source of truth for all constants. Edit here to change model, dataset,
output directory, or any training hyperparameter.

### `ctc_training/processor.py`
- `normalise_text(text)` — lowercase, strip punctuation except apostrophe,
  collapse whitespace.
- `build_vocab(dataset, vocab_path)` — iterates training transcripts, builds a
  character-level `vocab.json` with `|` (word boundary), `[UNK]`, `[PAD]`.
- `get_processor(vocab_path)` — constructs `Wav2Vec2Processor` from the vocab
  file. Must be called after `build_vocab`.

### `ctc_training/data.py`
- `load_speechocean()` — downloads dataset, resamples audio to 16 kHz,
  carves 10% validation split from train.
- `make_preprocess_fn(processor)` — returns a closure that converts a raw
  sample to `{input_values, attention_mask, labels}`. Samples longer than
  `MAX_DURATION_SEC` return `None` fields and are later filtered out.
- `prepare_dataset(dataset, processor)` — applies preprocessing via
  `dataset.map` and removes too-long rows.

### `ctc_training/collator.py`
`DataCollatorCTCWithPadding` — pads `input_values` and `labels` separately
(required by CTC). Label padding uses `-100` so loss ignores pad positions.

### `ctc_training/metrics.py`
`make_compute_metrics(processor)` — returns a function that decodes predicted
and reference token IDs and computes WER via `jiwer`.

### `ctc_training/model.py`
`build_model(processor)` — loads `HubertForCTC`, re-initialises the LM head
to match the vocabulary size, and freezes the CNN feature encoder.
To enable full fine-tuning, remove `model.freeze_feature_encoder()`.

---

## Extending to Stage 2 (Scoring Heads)

1. Load the saved encoder from `OUTPUT_DIR`.
2. Freeze the encoder weights.
3. Add regression heads on top for phoneme/word/utterance scores.
4. Train on SpeechOcean762 score annotations (`phones`, `words`, `utterance` columns).

---

## Common Issues

| Symptom | Fix |
|---|---|
| CUDA out of memory | Reduce `per_device_train_batch_size` to 2 or disable `gradient_checkpointing` |
| `vocab.json` mismatch | Delete `vocab.json` and rerun — it will be rebuilt |
| Slow preprocessing | Increase `num_proc` in `prepare_dataset` or cache with `dataset.save_to_disk` |
| WER not improving | Try unfreezing CNN encoder (`remove freeze_feature_encoder()`) or lowering LR |
