# Pronunciation Scoring

Automatic pronunciation scoring for L2 English learners.
The system uses a HuBERT encoder fused with reference phoneme embeddings via
cross-attention to predict five sentence-level scores and auxiliary prosodic features.

---

## Architecture

```
Audio waveform (16 kHz)
    → HuBERT encoder  (layer-weighted sum of all hidden states)
    → Linear projection  →  d_model = 256
    → Audio Transformer  (1 layer, pre-LN)
    → CrossAttentionFusion  (Q = audio frames, K/V = phoneme embeddings)  ←── Reference phonemes
    → Fusion Transformer  (2 layers, pre-LN)                                   from words[*]["phones"]
    → mean-pool over time
    → MLPScoringHead  →  5 scores  [0, 1]   (accuracy, completeness, fluency, prosody, total)
    → Prosody aux head  →  5 prosodic features
```

Phoneme source: **SpeechOcean762** `words[*]["phones"]` field — no G2P, no ASR alignment.

---

## Repository Structure

```
pronunciation-scoring/
├── models/
│   ├── phoneme_vocab.py      # ARPABET vocabulary (72 tokens), normalization helpers
│   ├── phoneme_embedder.py   # PhonemeEmbedder — token + positional embeddings
│   ├── scoring_heads.py      # CrossAttentionFusion, MLPScoringHead
│   ├── scoring_model.py      # HubertScoringModel — full end-to-end model
│   └── train.py              # Training entry point (TrainConfig, training loop, CLI)
├── utils/
│   └── dataset.py            # SpeechOcean762Dataset, prosody features, collate_fn
├── ctc_training/             # Legacy Stage-1 CTC fine-tuning pipeline
│   ├── config.py
│   ├── processor.py
│   ├── data.py
│   ├── collator.py
│   ├── metrics.py
│   └── model.py
├── inference/                # Inference utilities
├── train_hubert_speechocean.py  # Legacy Stage-1 entry point
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Key dependencies:** `torch`, `transformers`, `datasets`, `librosa`, `scipy`

> `librosa` is imported lazily inside `extract_prosody_features` only — the
> module loads without it when prosody extraction is not needed.

---

## Training

### Quick start

```bash
python -m models.train
```

This downloads `facebook/hubert-base-ls960` and the SpeechOcean762 dataset on
first run, then trains for 10 epochs saving to `runs/exp1/`.

### CLI options

```
python -m models.train --help

  --output-dir OUTPUT_DIR   Checkpoint and log directory  (default: runs/exp1)
  --epochs EPOCHS           Training epochs               (default: 10)
  --batch-size BATCH_SIZE   Training batch size           (default: 4)
  --lr LR                   Peak AdamW learning rate      (default: 2e-5)
  --weight-decay WEIGHT_DECAY                             (default: 0.01)
  --num-workers NUM_WORKERS DataLoader workers            (default: 2)
  --seed SEED               Random seed                   (default: 42)
  --no-fp16                 Disable AMP / FP16
  --model-name MODEL_NAME   HuggingFace HuBERT model ID
```

### Example runs

```bash
# Reproduce default experiment
python -m models.train --output-dir runs/exp1

# Longer run on a bigger GPU
python -m models.train --epochs 20 --batch-size 8 --output-dir runs/exp2

# CPU-only / debug
python -m models.train --no-fp16 --num-workers 0 --batch-size 2
```

### Outputs

Each run writes to `--output-dir`:

| File | Contents |
|---|---|
| `config.json` | Full `TrainConfig` snapshot |
| `best_checkpoint.pt` | Best model by Pearson correlation on `total` score |
| `metrics.csv` | Per-epoch loss + correlation table |

**Checkpoint format:**

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "config": dict,          # asdict(TrainConfig)
    "test_metrics": dict,    # pearson/spearman/mae per aspect
    "pcc_total": float,
}
```

---

## Key Design Decisions

### Phoneme vocabulary

`models/phoneme_vocab.py` defines a fixed 72-token ARPABET vocabulary:

| Group | Count | Example |
|---|---|---|
| Special (`<pad>`, `<unk>`, `<sil>`) | 3 | — |
| Vowels × 3 stress levels | 45 | `AH0`, `AH1`, `AH2` |
| Consonants (no stress) | 24 | `B`, `CH`, `SH` |

`normalize_phoneme_token` handles missing stress markers (defaults to `0`),
silence variants (`SP`, `SIL`, `<SIL>`), and unknown tokens gracefully.

### Layer-weighted HuBERT

All `num_hidden_layers + 1` hidden states (including the embedding output) are
combined via a learned softmax-weighted sum:

```python
weights = softmax(self.layer_weights)          # (13,) for HuBERT-base
weighted = (weights[:, None, None, None] * stacked).sum(0)
```

The last `num_unfreeze_hubert_layers` transformer layers remain trainable
(default: all 12). The CNN feature extractor is always frozen.

### Loss

Combined Huber (Smooth L1) loss:

```
total_loss = lambda_sent * huber(sent_pred, sent_scores)
           + lambda_prosody * huber(prosody_pred, z_score(prosody_feats))
```

Prosody targets are z-score normalised per batch so scale differences between
features do not dominate. Default `lambda_sent=1.0`, `lambda_prosody=0.5`.

### Learning rate schedule

Linear warmup (`warmup_ratio=0.1` of total steps) followed by linear decay to 0,
stepped once per batch.

---

## Dataset

**[mispeech/speechocean762](https://huggingface.co/datasets/mispeech/speechocean762)**
— 5,000 English utterances, human pronunciation scores at phoneme, word, and
utterance level.

Score keys used: `accuracy`, `completeness`, `fluency`, `prosody`, `total`
(each 0–10, normalised to 0–1 during training).

`utils/dataset.py` exposes:

- `SpeechOcean762Dataset(split, max_audio_seconds, insert_silence)` — PyTorch Dataset
- `collate_fn(batch)` — pads audio + phoneme sequences, emits attention/phoneme masks
- `extract_prosody_features(audio, sr)` — returns `[rms, zcr, peak_rate, pitch_mean, pitch_std]`

---

## Loading a Checkpoint

```python
import torch
from models.scoring_model import HubertScoringModel

ckpt = torch.load("runs/exp1/best_checkpoint.pt", map_location="cpu")
cfg  = ckpt["config"]

model = HubertScoringModel(
    model_name=cfg["model_name"],
    d_model=cfg["d_model"],
    num_heads=cfg["num_heads"],
    num_audio_transformer_layers=cfg["num_audio_transformer_layers"],
    num_fusion_transformer_layers=cfg["num_fusion_transformer_layers"],
    mlp_hidden_layers=cfg["mlp_hidden_layers"],
    dropout=cfg["dropout"],
    num_unfreeze_hubert_layers=cfg["num_unfreeze_hubert_layers"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

---

## Legacy: Stage-1 CTC Pre-training

The `ctc_training/` module and `train_hubert_speechocean.py` contain the earlier
CTC fine-tuning pipeline (HuBERT-Large on SpeechOcean762 transcripts).  It is
kept for reference and checkpoint comparison; new experiments use `models/train.py`.

```bash
python train_hubert_speechocean.py   # produces hubert-large-speechocean-ctc/
```
