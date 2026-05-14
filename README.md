# Pronunciation Scoring

Automatic pronunciation scoring for L2 English learners.
The system uses a HuBERT encoder fused with a BGE sentence embedding via
cross-attention to predict five sentence-level scores and auxiliary prosodic features.

---

## Architecture

```
Audio waveform (16 kHz)
    → HuBERT encoder  (layer-weighted sum of all hidden states)
    → Linear projection  →  d_model = 256
    → Audio Transformer  (1 layer, pre-LN)
    → CrossAttentionFusion  (Q = audio frames, K/V = text embedding)  ←── Reference script
    → Fusion Transformer  (2 layers, pre-LN)                               BGE mean-pool vector
    → mean-pool over time
    → MLPScoringHead  →  5 scores  [0, 1]   (accuracy, completeness, fluency, prosody, total)
    → Prosody aux head  →  5 prosodic features
```

Text branch: **BAAI/bge-small-en-v1.5** — script tokenised and mean-pooled to a single
`(B, 1, 256)` sentence vector. BGE encoder frozen by default.

---

## Repository Structure

```
pronunciation-scoring/
├── models/
│   ├── text_embedder.py      # BGETextEmbedder — wraps BGE, projects 384 → d_model
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

Downloads `facebook/hubert-base-ls960`, `BAAI/bge-small-en-v1.5`, and the
SpeechOcean762 dataset on first run, then trains for 10 epochs saving to `runs/bge_exp1/`.

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `runs/bge_exp1` | Checkpoint and log directory |
| `--epochs` | `10` | Training epochs |
| `--batch-size` | `4` | Training batch size |
| `--lr` | `2e-5` | Peak AdamW learning rate |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--num-workers` | `2` | DataLoader worker processes |
| `--seed` | `42` | Random seed |
| `--no-fp16` | — | Disable AMP / FP16 |
| `--hubert-model-name` | `facebook/hubert-base-ls960` | HuggingFace HuBERT model ID |
| `--bge-model-name` | `BAAI/bge-small-en-v1.5` | HuggingFace BGE model ID |
| `--no-freeze-bge` | — | Unfreeze BGE encoder weights during training |

### Example runs

```bash
# Reproduce default experiment
python -m models.train --output-dir runs/bge_exp1

# Longer run on a bigger GPU
python -m models.train --epochs 20 --batch-size 8 --output-dir runs/bge_exp2

# Fine-tune BGE encoder as well (use carefully — risk of catastrophic forgetting)
python -m models.train --no-freeze-bge --output-dir runs/bge_unfrozen

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

### BGE text branch

`models/text_embedder.py` wraps `BAAI/bge-small-en-v1.5` (hidden size 384):

- **Mean pooling** (not CLS): BGE is trained with mean pooling as its pooling strategy.
- **Single sentence vector** `(B, 1, 256)`: sentence-level conditioning matches the
  original paper design; keeps cross-attention lightweight with no padding concerns.
- **Frozen by default**: BGE already has strong sentence representations; fine-tuning
  on the small SpeechOcean762 dataset (5,000 utterances) risks catastrophic forgetting.

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

- `SpeechOcean762Dataset(split, max_audio_seconds, bge_model_name, max_text_length)` — PyTorch Dataset
- `collate_fn(batch)` — pads audio + text token sequences, emits audio/text attention masks
- `extract_prosody_features(audio, sr)` — returns `[rms, zcr, peak_rate, pitch_mean, pitch_std]`

---

## Loading a Checkpoint

```python
import torch
from models.scoring_model import HubertScoringModel

ckpt = torch.load("runs/bge_exp1/best_checkpoint.pt", map_location="cpu")
cfg  = ckpt["config"]

model = HubertScoringModel(
    hubert_model_name=cfg["hubert_model_name"],
    bge_model_name=cfg["bge_model_name"],
    d_model=cfg["d_model"],
    num_heads=cfg["num_heads"],
    num_audio_transformer_layers=cfg["num_audio_transformer_layers"],
    num_fusion_transformer_layers=cfg["num_fusion_transformer_layers"],
    mlp_hidden_layers=cfg["mlp_hidden_layers"],
    dropout=cfg["dropout"],
    num_unfreeze_hubert_layers=cfg["num_unfreeze_hubert_layers"],
    freeze_bge=cfg["freeze_bge"],
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
