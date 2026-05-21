# Phone-level Modeling Pipeline

## 🎯 Overview

This module implements **phoneme-level pronunciation assessment** using:

* A **phoneme-level HuBERT CTC model** (pretrained separately)
* **Forced alignment**
* **GOP (Goodness of Pronunciation)**
* A **Transformer-based regression model**

The goal is to predict:

```text
phoneme-level accuracy score ∈ [0, 2]
```

for each phoneme in an utterance.

---

## 🧠 Pipeline Architecture

```text
Audio
  ↓
HuBERT (phoneme CTC)
  ↓
Frame-level outputs:
    - hidden states (SSL features)
    - log_probs (phoneme posterior)
  ↓
CTC Forced Alignment
  ↓
Frame → Phoneme mapping
  ↓
Feature extraction:
    - SSL (aggregated)
    - GOP
    - duration
    - phoneme embedding
  ↓
Transformer Encoder
  ↓
Phoneme-level accuracy prediction
```

---

## 📦 Input Requirements

Each sample from **Speechocean762** must provide:

* `audio`: waveform (16kHz)
* `words[*]["phones"]`: phoneme sequence (with stress markers)
* `words[*]["phones-accuracy"]`: ground-truth scores

---

## 🔤 Phoneme Representation

* Uses ARPAbet phonemes (e.g., `AA0`, `IH1`)
* **Stress markers are preserved**
* No G2P is required (dataset already provides phonemes)

---

## ⚙️ Step-by-Step Pipeline

---

### 1. Feature Extraction (HuBERT)

Using a trained phoneme-level CTC model:

```python
ssl, log_probs = extract_ssl_and_logprob(model, input_values)
```

Outputs:

* `ssl`: (T_frame, 1024)
* `log_probs`: (T_frame, vocab_size)

---

### 2. Forced Alignment

Align frames to phonemes:

```python
frame2phone = align(log_probs, phone_ids)
```

Output:

* `frame2phone`: (T_frame,) → index of phoneme per frame

---

### 3. Frame → Phoneme Aggregation

Aggregate SSL features:

```python
phone_ssl = aggregate_ssl(ssl, frame2phone, num_phones)
```

Output:

* `phone_ssl`: (T_phone, 1024)

---

### 4. GOP (Goodness of Pronunciation)

```python
gop = compute_gop(log_probs, phone_ids, frame2phone)
```

Definition:

```text
GOP = log P(target phoneme) − max log P(other phonemes)
```

Output:

* `gop`: (T_phone, 1)

---

### 5. Duration Feature

```python
dur = compute_duration(frame2phone, num_phones)
```

* Count frames per phoneme
* Apply log scaling

Output:

* `dur`: (T_phone, 1)

---

### 6. Phoneme Embedding

Trainable embedding:

```python
embedding = nn.Embedding(num_phones, 64)
```

Output:

* `phone_embed`: (T_phone, 64)

---

### 7. Feature Fusion

Concatenate all features:

```text
[SSL (1024) | GOP (1) | duration (1) | embedding (64)]
```

Final shape:

```text
(T_phone, 1090)
```

---

### 8. Transformer Model

* Input projection: 1090 → 256
* 4-layer Transformer Encoder
* Positional encoding

Output:

```text
(T_phone, 1)
```

→ predicted phoneme accuracy

---

### 9. Loss Function

Masked Mean Squared Error:

```python
loss = ((pred - target)**2 * mask).sum() / mask.sum()
```

* Ignores padding positions

---

## 🧪 Training Setup

* Optimizer: Adam
* Learning rate: 1e-4
* Batch size: 2–8 (depends on GPU)
* Epochs: 20+

---

## ⚠️ Critical Implementation Notes

### 1. Alignment Quality

* Poor CTC → bad alignment → unusable features
* Ensure **PER < 20%** for CTC model

---

### 2. GOP Stability

* Must be normalized:

```python
gop = (gop - mean) / std
```

* Check distribution is not constant

---

### 3. Padding Mask

Transformer must use:

```python
src_key_padding_mask
```

---

### 4. Embedding Placement

* MUST be inside model (not dataset)
* Otherwise it won’t be trained

---

### 5. Vocabulary Consistency

Ensure:

```text
CTC vocab == dataset phonemes
```

Mismatch → invalid GOP

---

## 📊 Output

For each utterance:

```text
[T_phone] → predicted phoneme scores
```

---

## 🚀 Extensions (for research)

* Replace Transformer → Conformer
* Add word-level modeling
* Hierarchical structure (phone → word → utterance)
* Ordinal regression loss
* Use mispronunciation labels

---

## ✅ Summary

This pipeline enables:

* Fully supervised phoneme-level scoring
* Integration of acoustic + linguistic features
* End-to-end training on real CAPT dataset

---

## 👨‍🔬 Research Positioning

Compared to prior work:

* Uses **strong phoneme supervision** (vs weak in HiPPO)
* Explicit **GOP modeling**
* Modular and extensible architecture
