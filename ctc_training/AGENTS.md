# AGENT.md — Phoneme-level HuBERT CTC Training

## 🎯 Objective

Train a HuBERT-based CTC model that predicts **phoneme sequences (with stress markers)** instead of characters.

This model will later be used for:

* Forced alignment
* GOP computation

---

## 🔥 Key Changes from Current Setup

### ❌ Current (WRONG for CAPT)

* Character-level vocabulary
* WER metric
* Text normalization

### ✅ New (REQUIRED)

* Phoneme-level vocabulary (ARPAbet with stress)
* PER (Phoneme Error Rate)
* No text normalization

---

## 📦 Dataset: Speechocean762

Use:

* `sample["audio"]`
* `sample["words"][i]["phones"]`

DO NOT use:

* `sample["text"]`

---

## 🔤 Phoneme Vocabulary

### Build from dataset:

* Collect all phonemes from:

  ```
  sample["words"][i]["phones"]
  ```

* Keep stress markers:

  ```
  AA0, AA1, AA2 → MUST remain distinct
  ```

---

### Add special tokens:

| Token   | Purpose |               |
| ------- | ------- | ------------- |
| `       | `       | word boundary |
| `[PAD]` | padding |               |
| `[UNK]` | unknown |               |

---

## 🧠 Model

Use:

```
HubertForCTC
```

Changes:

* `vocab_size = len(phoneme_vocab)`
* `pad_token_id = tokenizer.pad_token_id`

---

## 🔒 Freeze Strategy

Keep:

```
model.freeze_feature_encoder()
```

Optional later:

* unfreeze for full fine-tuning

---

## 📥 Input / Output

### Input:

```
audio waveform
```

### Target:

```
phoneme sequence (with | between words)
```

Example:

```
["M", "AA1", "R", "K", "|", "IH0", "T"]
```

---

## 📉 Loss

Standard CTC loss:

```
loss = model(input_values, labels=labels).loss
```

---

## 📏 Evaluation Metric

Replace WER with PER:

```
PER = edit_distance(phoneme_seq_pred, phoneme_seq_gt)
```

---

## ⚠️ Critical Constraints

### 1. Vocabulary MUST match dataset phones

No mapping allowed:

```
AA1 ≠ AA0
```

---

### 2. Sequence formatting

* Insert "|" between words
* No trailing "|"

---

### 3. Tokenizer consistency

Tokenizer must be built ONLY from phoneme vocab

---

## 🧪 Training Tips

* Batch size: as large as GPU allows
* LR: 1e-4 → 3e-5
* Warmup: 10%
* Epochs: 20–50

---

## ✅ Success Criteria

* PER < 20% (acceptable)
* Alignment visually correct
* No collapse in predictions

---

## 🚫 Common Mistakes

❌ Using character vocab
❌ Removing stress markers
❌ Mixing text + phoneme labels
❌ Wrong tokenizer

---

## 📤 Output

Save:

* model checkpoint
* tokenizer (phoneme vocab)

These will be used in:
→ phone-level modeling pipeline

---
