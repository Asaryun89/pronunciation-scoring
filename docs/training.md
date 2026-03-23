# Training Process — HuBERT Multi-Task on SpeechOcean762

## Quick Start

```bash
python models/train.py \
  --dataset mispeech/speechocean762 \
  --epochs 10 \
  --batch_size 4 \
  --out_dir ckpt_hubert_multitask
```

Full options:

```bash
python models/train.py \
  --dataset mispeech/speechocean762 \
  --train_split train \
  --valid_split test \
  --model facebook/hubert-base-ls960 \
  --out_dir ckpt_hubert_multitask \
  --epochs 10 \
  --batch_size 4 \
  --lr 2e-5 \
  --wd 0.01 \
  --dropout 0.1 \
  --freeze_fe \
  --max_words 60 \
  --max_phones 10 \
  --num_workers 2 \
  --seed 42 \
  --w_sent 1.0 \
  --w_pfeat 0.5 \
  --w_words 1.0 \
  --w_phone 0.5 \
  --log_every 50
```

Checkpoint is written to `ckpt_hubert_multitask/best.pt` (best `pearson_total` on the test split).

---

## Dataset: SpeechOcean762

| Property | Value |
|---|---|
| HuggingFace slug | `mispeech/speechocean762` |
| Utterances | ~5000 |
| Speakers | 250 (L2 English learners) |
| Train split | `train` |
| Test split  | `test` |

### Annotation schema (per utterance)

| Field | Type | Scale | Description |
|---|---|---|---|
| `accuracy` | int | 0–10 | Phoneme accuracy |
| `completeness` | float | 0–10 | Completeness of utterance |
| `fluency` | int | 0–10 | Speaking fluency |
| `prosodic` | int | 0–10 | Prosody quality |
| `total` | int | 0–10 | Overall pronunciation score |
| `words[].total` | int | 0–10 | Word-level overall score |
| `words[].phones-accuracy` | list[float] | 0–2 | Per-phone accuracy |

---

## Score Constants

Defined in `models/constants.py`, shared between training and inference:

```python
SENT_DIMS   = ["total", "accuracy", "fluency", "prosodic", "completeness"]
SENT_SCALE  = 10.0   # utterance + word labels are 0–10
PHONE_SCALE =  2.0   # phones-accuracy is 0–2
PROSODY_DIMS = ["rms", "zcr", "peak_rate", "f0_mean", "f0_std"]
```

All model heads output values in `[0, 1]` via sigmoid. Multiply by the appropriate scale to recover display values.

---

## Pipeline Step by Step

### 1. Startup (`main()` in `models/train.py`)

```
set_seed(42)
create_accelerator()            ← accelerate lib or SimpleAccelerator fallback
load_dataset("mispeech/speechocean762")
cast_column("audio", Audio(decode=False))   ← keep bytes/path, defer decode
preflight schema check on first 32 rows (train + valid)
Wav2Vec2FeatureExtractor.from_pretrained(hubert_name)
Collator × 2 (train, valid)
DataLoader × 2
HubertMultiTask(...)
AdamW optimizer (lr=2e-5, weight_decay=0.01)
accelerator.prepare(model, optimizer, loaders)
```

`Audio(decode=False)` prevents HuggingFace from immediately decoding audio arrays. Decoding happens inside `Collator` at batch time, giving full control over resampling and error reporting.

The preflight check validates that the first 32 rows match the expected schema before any GPU work begins, catching annotation issues early.

---

### 2. Data Collation (`data/collate.py` → `Collator`)

Called once per batch by `DataLoader` with a list of raw dataset rows.

#### Step A — Audio decode & prosody

```
row.audio (bytes/path)
  → decode_audio_from_example()   ← soundfile read + resample to 16kHz
  → peak-normalize (÷ max amplitude)
  → basic_prosody_features()      ← rms, zcr, peak_rate, f0_mean, f0_std
  → prosody_to_vector()           → shape (5,)
```

Peak-normalizing before prosody extraction makes features amplitude-invariant.

#### Step B — HuBERT feature extraction

```
list[audio arrays]
  → Wav2Vec2FeatureExtractor (padding=True)
  → input_values (B, T_samples)
  → attention_mask (B, T_samples)
```

#### Step C — Label normalization

All annotation labels are divided by their scale so model targets are in `[0, 1]`:

| Label | ÷ Scale | Range |
|---|---|---|
| utterance scores (5 dims) | ÷ 10 | [0, 1] |
| `words[].total` | ÷ 10 | [0, 1] |
| `words[].phones-accuracy` | ÷ 2 | [0, 1] |

#### Step D — Padded output tensors

```
sent_targets  (B, 5)       — 5 utterance scores
prosody_feats (B, 5)       — 5 audio-computed features
word_scores   (B, W)       — word totals; zero-padded
word_mask     (B, W)       — True where a real word exists
phone_scores  (B, W, P)    — phone accuracies; zero-padded
phone_mask    (B, W, P)    — True where a real phone exists
```

`W = min(max_words=60, max words in batch)`, `P = max_phones_per_word=10`.

---

### 3. Model Forward Pass (`HubertMultiTask.forward()`)

```
input_values (B, T_samples)
    ↓  HuBERT (facebook/hubert-base-ls960)
hidden (B, T_frames, 768)        ← ~50 frames/sec

    ↓  mean over all T_frames
sent_emb (B, 768)
    ├── sentence_head + sigmoid  → sent_pred  (B, 5)   — utterance scores [0,1]
    └── prosody_feat_head        → prosody_pred (B, 5)  — audio features (unbounded)

    ↓  even-split per word (training baseline)
    for each word wi in [0, valid_w):
        frames[ws:we] → mean pool → word_head + sigmoid → word_pred[b, wi]
        for each phone pi in [0, valid_p):
            frames[ps:pe] → mean pool → phone_head + sigmoid → phone_pred[b, wi, pi]
```

Word/phone frame spans are divided evenly across `T_frames` during training (no ASR needed per batch). At inference, actual ASR timestamps replace the even splits via `forward_inference()`.

---

### 4. Loss Computation (`compute_loss()`)

Four terms, all using **Smooth L1 (Huber) loss**:

| Term | Compared | Default weight |
|---|---|---|
| `sent_loss` | `sent_pred` vs `sent_targets` | 1.0 |
| `pfeat_loss` | `prosody_pred` vs `prosody_feats` | 0.5 |
| `word_loss` | `word_pred[mask]` vs `word_scores[mask]` | 1.0 |
| `phone_loss` | `phone_pred[mask]` vs `phone_scores[mask]` | 0.5 |

```
total_loss = w_sent·sent + w_pfeat·pfeat + w_words·word + w_phone·phone
```

Masks are applied to `word_loss` and `phone_loss` so padding zeros never contribute gradients.

Smooth L1 is chosen over MSE for robustness: it behaves like L2 near zero (smooth gradients) and like L1 for large errors (outlier-resistant).

---

### 5. Training Loop

```
for epoch in 1..N:
    model.train()
    for step, batch in train_loader:
        move batch → device
        model.forward() → outputs
        compute_loss() → total_loss
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)   ← frees gradient memory
        print every 50 steps: total / sent / word / phone losses

    ↓ validation
    eval_epoch() → metrics dict
    if pearson_total > best so far:
        save ckpt_hubert_multitask/best.pt
```

---

### 6. Validation & Metrics (`eval_epoch()`)

Runs under `@torch.no_grad()`. Collects all `sent_pred` + `sent_targets` across the full validation set, then computes per-dimension metrics:

| Metric | Meaning |
|---|---|
| `mae_{dim}` | Mean absolute error in [0,1] scale |
| `rmse_{dim}` | Root mean squared error |
| `pearson_{dim}` | Linear correlation with ground truth |
| `spearman_{dim}` | Rank correlation (robust to non-linear effects) |

**Best checkpoint criterion**: `pearson_total` — Pearson correlation on the `total` score. This matches the evaluation standard used in the SpeechOcean762 paper.

---

### 7. Checkpoint

`best.pt` is saved whenever `pearson_total` improves. It contains only the model `state_dict` (no optimizer state). Load at inference via:

```python
cfg = PredictorConfig(checkpoint_path="ckpt_hubert_multitask/best.pt")
predictor = PronunciationPredictor(cfg)
```

---

## Full Data Flow Diagram

```
SpeechOcean762 row
  ├─ audio.bytes/path ──→ decode → resample 16kHz → normalize
  │                                  ↓                   ↓
  │                          Wav2Vec2FE           prosody_features (5,)
  │                          input_values (B,T)   prosody_feats (B,5)
  │
  ├─ accuracy/fluency/prosodic/completeness/total  ÷10  → sent_targets  (B,5)
  ├─ words[].total                                 ÷10  → word_scores   (B,W)
  └─ words[].phones-accuracy[]                     ÷2   → phone_scores  (B,W,P)
                 │
                 ▼
        HubertMultiTask.forward()
                 │
      HuBERT hidden states (B, T_frames, 768)
                 │
        ┌────────┴──────────────────────────────────┐
        │ mean pool all frames                      │ even-split per word
        │ (B, 768)                                  │ (W × mean pool → 768 each)
        │                                           │
   sentence_head   prosody_head                word_head    phone_head
   sent_pred (B,5) prosody_pred (B,5)          word_pred(B,W) phone_pred(B,W,P)
                 │
         compute_loss()
      SmoothL1 × 4 weighted terms
                 │
      accelerator.backward()  →  optimizer.step()
                 │
         eval_epoch()  (after each epoch)
      MAE / RMSE / Pearson / Spearman per dimension
                 │
      save best.pt  on best pearson_total
```

---

## Module Map

| File | Responsibility |
|---|---|
| `models/constants.py` | Shared score constants (SENT_DIMS, scales) |
| `data/validate.py` | Schema validation + audio decode helpers |
| `data/collate.py` | `Collator` — batch assembly for DataLoader |
| `models/train.py` | `HubertMultiTask`, `compute_loss`, `eval_epoch`, `main()` |
| `inference/predictor.py` | End-to-end inference using trained checkpoint |
| `utils/prosody.py` | `basic_prosody_features`, `prosody_to_vector` |
| `utils/alignment.py` | `build_word_segments` — seconds → frame indices |

---

## Known Limitations

| Issue | Impact |
|---|---|
| Even frame splits during training | Word/phone head quality is lower than utterance head |
| No forced aligner at inference | `phones`, `phones-accuracy`, `stress` fields are empty in output |
| `pyworld` optional | F0 features fall back to 0.0 if not installed |
| `utils/fusion.py` `fuse_scores()` | Defined but unused in the active inference path |
