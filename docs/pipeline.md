# Pipeline Stages

This document walks through the pronunciation-scoring pipeline as a sequence of
processing stages, from raw dataset to trained scoring model. For the file/module
layout see `README.md`; for the full reproducibility guide see `AGENTS.md`.

```
SpeechOcean762 (HF dataset)
        │
        ▼
┌───────────────────────────┐     ┌──────────────────────────────────────┐
│ Stage 1 — stage1_ctc/      │ ──▶ │ Stage 2 — stage2_scoring/             │
│ CTC fine-tuning            │     │ Multitask scoring (audio + text)      │
│ → outputs/hubert-large-    │     │ → outputs/ckpt_hubert_multitask/      │
│   speechocean-ctc/         │     │   best.pt, result.csv                 │
└───────────────────────────┘     └──────────────────────────────────────┘
```

Stage 2 can start from either the Stage 1 checkpoint (`--backbone stage1`) or
the stock pretrained encoder (`--backbone hubert-large`) — Stage 1 is an
optional warm-start, not a hard dependency.

---

## Stage 0 — Dataset

**Source:** [`mispeech/speechocean762`](https://huggingface.co/datasets/mispeech/speechocean762)
on the HuggingFace Hub — ~5,000 English utterances from 250 L2 learners, with
`train` / `test` splits and human-annotated pronunciation scores.

Per-utterance fields used downstream:

| Field | Scale | Used by |
|---|---|---|
| `audio` | 16 kHz waveform (bytes/path) | both stages |
| `text` | reference transcript | both stages (CTC labels / text stream) |
| `accuracy`, `fluency`, `prosodic`, `total`, `completeness` | 0–10 | Stage 2 targets (`SENT_DIMS`) |
| `words[].*` | 0–10 / 0–2 | not currently consumed (see "Known gaps") |

---

## Stage 1 — CTC Pretext Fine-tuning (`stage1_ctc/`)

**Goal:** fine-tune `facebook/hubert-large-ls960-ft` to predict the transcript
via CTC, producing a speech encoder whose representations are tuned to this
dataset's speakers/accents before Stage 2 reuses it.

Pipeline (`stage1_ctc/train.py main()`):

1. **Load & split** — `data.py: load_speechocean()` downloads the dataset, casts
   `audio` to 16 kHz, and carves a 10% validation split from `train`.
2. **Build vocabulary** — `processor.py: build_vocab()` normalises every training
   transcript (`normalise_text`: lowercase, strip punctuation except apostrophe,
   collapse whitespace), collects the character set, and writes
   `outputs/vocab.json` (adds `|` word-boundary, `[UNK]`, `[PAD]`). Skipped if the
   file already exists.
3. **Build processor** — `processor.py: get_processor()` wraps the vocab in a
   `Wav2Vec2CTCTokenizer` + `Wav2Vec2FeatureExtractor` → `Wav2Vec2Processor`.
4. **Preprocess dataset** — `data.py: prepare_dataset()` maps each example through
   the processor (`input_values`, `attention_mask`) and tokenises the normalised
   transcript into `labels`; utterances longer than `MAX_DURATION_SEC=20s` are
   dropped.
5. **Build model** — `model.py: build_model()` loads `HubertForCTC`, re-initialises
   the LM head to the new vocab size, and freezes the CNN feature extractor
   (`freeze_feature_encoder()`).
6. **Collate** — `collator.py: DataCollatorCTCWithPadding` pads `input_values` and
   `labels` independently (CTC requirement); label padding uses `-100` so it's
   ignored by the loss.
7. **Train** — HuggingFace `Trainer` with `TRAINING_ARGS` from `config.py`:
   30 epochs, batch 16 × grad-accum 4, lr 1e-4 linear schedule with 10% warmup,
   fp16 + gradient checkpointing, best checkpoint selected by **val WER**.
8. **Evaluate** — `metrics.py: make_compute_metrics()` decodes predicted vs.
   reference token IDs and computes **WER** via `jiwer`.
9. **Save** — model + processor → `outputs/hubert-large-speechocean-ctc/`, plus
   a final test-set WER in `test_results.json`.

**Run:** `python -m stage1_ctc.train`

---

## Stage 2 — Multitask Scoring Model (`stage2_scoring/`)

**Goal:** train `HubertMultiTask` to predict 5 utterance-level pronunciation
scores (`total, accuracy, fluency, prosodic, completeness`) from audio plus the
reference transcript.

### 2.1 — Backbone resolution

`train.py main()` picks the HuBERT checkpoint to start from:

- `--model <path>` — explicit checkpoint (overrides everything else)
- `--backbone stage1` — Stage 1 output (`outputs/hubert-large-speechocean-ctc/`)
- `--backbone hubert-large` (default) — stock `facebook/hubert-large-ll60k`

### 2.2 — Dataset load & validation

- `load_dataset(...)`, cast `audio` with `Audio(decode=False)` so raw bytes/path
  are kept (decoding happens per-batch in the collator, not eagerly).
- `validate.py: validate_example_schema()` checks the first 32 rows of each
  split against the expected SpeechOcean762 schema (utterance numeric fields,
  `words[]` structure, audio payload) — fails fast before any GPU work.

### 2.3 — Batch collation (`collate.py: Collator`)

Run once per batch by the `DataLoader`:

1. **Decode audio** — read bytes/path with `soundfile`, mono-mix, then
   `utils.preprocessing`: `resample` → 16 kHz → `peak_normalize` → `vad_trim`
   (webrtcvad) → `peak_normalize` again. Anything shorter than 640 samples
   after trimming is zero-padded (HuBERT's CNN minimum).
2. **Prosody targets** — `utils.prosody.basic_prosody_features()` (rms, zcr,
   peak-rate, F0 mean/std via `pyworld` if installed, else 0) →
   `prosody_to_vector()` → fixed-range `(5,)` vector, computed on the
   peak-normalised (pre zero-mean/unit-std) audio.
3. **Model input tensors** — each waveform is zero-mean/unit-std normalised,
   then zero-padded to the batch max length → `input_values (B,T)`,
   `attention_mask (B,T)`.
4. **Label tensors** — `sent_targets (B,5)` = `[total, accuracy, fluency,
   prosodic, completeness]` ÷ `SENT_SCALE (10.0)`, i.e. scaled to `[0,1]`.
5. **Text tokenisation** (if `--text_model` set) — reference `text` →
   `text_input_ids`, `text_attention_mask` via the configured tokenizer
   (default `Qwen/Qwen3-Embedding-0.6B`).

### 2.4 — Model forward (`hubert_multitask.py: HubertMultiTask`)

**Audio path** — two interchangeable backbones, selected by `--use_multires_hubert`:

- *Default (single-stream)*: HuBERT → softmax-weighted sum over all 25 hidden
  states (`layer_weights`, learnable) → `Linear(H→d_model)` (`audio_proj`).
- *MultiResHuBERT* (`multires_hubert.py`, opt-in): `num_res_streams` independent
  Gaussian-initialised layer-weight vectors each produce their own weighted sum
  and `Linear(H→d_model)` projection; the streams are concatenated and fused
  back to `d_model` with one more `Linear`. See `notebooks/multires_hubert.ipynb`.

Both paths produce `audio_emb (B,T,d_model)` and feed the same downstream pipeline:

1. `--num_unfreeze_hubert_layers` (default 12) controls how many of HuBERT-Large's
   top 24 transformer layers are trainable; the rest (and the CNN extractor, if
   `--freeze_fe`) stay frozen.
2. **Pre-fusion self-attention** — optional `TransformerEncoder` over `audio_emb`
   (`--num_audio_transformer_layers`, default 1).
3. **Text path** (if enabled) — frozen (or fine-tuned) text encoder → mask-aware
   mean pool → `Linear(text_H→d_model) + LayerNorm` → `(B,1,d_model)` as K/V.
4. **Cross-attention fusion** (`scoring_heads.py: CrossAttentionFusion`) — audio
   is the query, text is K/V, residual + LayerNorm.
5. **Post-fusion Transformer** — `--num_transformer_layers` (default 2), pre-LN.
6. **Pooling** — mean over time → `utt_emb (B,d_model)`.
7. **Heads**:
   - `scoring_heads.py: MLPScoringHead` → `(B,5)` sigmoid scores (`sent_pred`)
   - `prosody_feat_head` (aux, training regulariser) → `(B,5)` unbounded
     prediction (`prosody_pred`)

`forward_inference()` is the inference-time variant: it maps ASR word
timestamps to frame spans (`utils/alignment.py: build_word_segments`) and can
append phone-level pooled embeddings to the text K/V sequence if a CTC phone
aligner provides spans — see "Known gaps" below.

### 2.5 — Loss (`train.py: compute_loss`)

```
sent_loss  = MSE(sent_pred[:, ACTIVE_SENT_IDXS], sent_targets[:, ACTIVE_SENT_IDXS])
pfeat_loss = MSE(prosody_pred, prosody_feats)
total_loss = w_sent * sent_loss + w_pfeat * pfeat_loss
```

`ACTIVE_SENT_IDXS = [total, accuracy, fluency, prosodic]` — `completeness` (idx 4)
is excluded because SpeechOcean learners almost always score 10/10, which would
collapse that head to a constant.

### 2.6 — Training loop

- Optimiser: `AdamW(lr, weight_decay)`; LR schedule: linear warmup
  (`--warmup_steps`, default 100) then constant.
- `create_accelerator()` uses HuggingFace `accelerate` if installed, else a
  `SimpleAccelerator` shim (single device, no-op `gather`/`prepare`).
- Per-step: forward → `compute_loss` → `accelerator.backward` → optimiser/scheduler
  step. Every `--log_every` steps, logs are printed and appended to
  `outputs/ckpt_hubert_multitask/result.csv`.

### 2.7 — Validation (`train.py: eval_epoch`)

Runs under `@torch.no_grad()` over the `test` split (default `--valid_split`).
For each of the 5 `SENT_DIMS`: `mae`, `rmse`, `pearson`, `spearman` against the
ground-truth `sent_targets`.

### 2.8 — Checkpointing & early stopping

- `best.pt` (state dict only) is saved whenever `pearson_total` improves.
- `--patience` (default 20) epochs without improvement triggers early stopping.
- If there's no validation split, every epoch is saved as `epoch_N.pt` instead.

**Run:**

```bash
python -m stage2_scoring.train --backbone stage1                       # warm-started from Stage 1
python -m stage2_scoring.train --backbone hubert-large                 # stock encoder
python -m stage2_scoring.train --backbone stage1 --use_multires_hubert # multi-resolution audio encoder
```

---

## Shared utilities (`utils/`)

| Module | Used for |
|---|---|
| `preprocessing.py` | `resample`, `peak_normalize`, `vad_trim`, `preprocess_wav` — audio normalisation chain shared by Stage 2 collation and (eventual) inference |
| `prosody.py` | `basic_prosody_features`, `prosody_to_vector` — auxiliary prosody regression targets |
| `alignment.py` | `build_word_segments` — seconds → frame-index spans for `forward_inference()` |

---

## Outputs (`outputs/`, gitignored)

| Path | Produced by | Contents |
|---|---|---|
| `vocab.json` | Stage 1 | character vocabulary |
| `hubert-large-speechocean-ctc/` | Stage 1 | fine-tuned `HubertForCTC` + processor + `test_results.json` (WER) |
| `ckpt_hubert_multitask/` | Stage 2 | `best.pt`, `result.csv` (per-step/epoch train+val metrics) |

---

## Known gaps

- `forward_inference()` (word-span / phone-span fusion) exists on the model but
  there is no standalone `inference`/`predictor` entry point in this repo yet —
  it is exercised only via direct calls.
- SpeechOcean762's per-word/per-phone annotations (`words[].*`) are validated by
  `validate.py` but not yet used as training targets — only the 5 utterance-level
  scores and the audio-derived prosody vector are trained on.
- `completeness` is predicted (head outputs 5 values) but excluded from the loss
  via `ACTIVE_SENT_IDXS`.
