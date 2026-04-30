# AGENT.md — Phone-level Modeling (CAPT)

## 🎯 Objective

Train a model to predict:

```
phoneme-level pronunciation accuracy
```

Using:

* HuBERT features
* GOP
* duration
* phoneme embedding

---

## 📦 Input Pipeline

### Step 1: HuBERT (phoneme CTC)

→ extract:

* hidden states (SSL features)
* log_probs (for GOP)

---

### Step 2: Forced Alignment

Use:

```
torchaudio CTC alignment
```

Output:

```
frame → phoneme mapping
```

---

### Step 3: Feature Construction

For each phoneme:

```
[SSL (1024) | GOP (1) | duration (1) | embedding (64)]
```

Final:

```
dim = 1090
```

---

## 🧠 Model

Transformer Encoder:

* d_model = 256
* layers = 4
* heads = 4

---

## 📤 Output

```
(B, T_phone)
```

→ predicted phoneme accuracy

---

## 📉 Loss

Masked MSE:

```
only compute loss on real phonemes
ignore padding
```

---

## 🧪 Training Tips

* Normalize GOP
* Log-scale duration
* Use masking correctly

---

## ⚠️ Critical Points

### 1. Alignment must be correct

Bad alignment → useless model

---

### 2. GOP distribution

Must NOT be constant

---

### 3. Padding

Transformer MUST receive:

```
src_key_padding_mask
```

---

## ✅ Success Criteria

* Loss decreases smoothly
* Predictions correlate with labels
* No NaNs

---

## 🚫 Common Mistakes

❌ Not masking padding
❌ Wrong alignment
❌ Unnormalized GOP
❌ Mismatched vocab

---

## 📈 Future Extensions

* Conformer
* Word-level modeling
* Hierarchical attention
* Ordinal loss

---
