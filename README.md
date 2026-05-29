# Multi-resolution HuBERT for Pronunciation Assessment

A PyTorch implementation of Multi-resolution HuBERT fine-tuned on the
[Speechocean762](https://www.openslr.org/101/) dataset for automatic
pronunciation assessment.

## Architecture

```
  s  ─── raw waveform (16 kHz)
  │
  f₀  Convolutional Feature Extractor  →  H̃₀  (masked CNN features)
  │
  f₁  High-Resolution Transformer Encoder  (layers 0–3, frozen)
  │
  ▼  DOWN  depthwise-sep strided conv  (stride = 2)
  │
  f₂  Low-Resolution Transformer Encoder   (layers 4–7)
  │
  ▲  UP    transposed conv + gated skip from f₁
  │
  f₃  High-Resolution Transformer Encoder  (layers 8–11)
  │        │
  g^q_R1   g^q_R2   ← unit-prediction heads (pre-training only)
  │
  PronunciationScoreHead
  ├─ mean-pool H₃   →  hi_proj
  └─ mean-pool H₂   →  lo_proj
         ↓ MLP ↓
  [accuracy | fluency | completeness | prosodic | total]
```

---

## Requirements

```bash
pip install -r requirements.txt
# Optional: enable wandb or tensorboard logging
pip install wandb          # or
pip install tensorboard
```

**requirements.txt** installs:
`torch ≥ 2.1`, `torchaudio ≥ 2.1`, `transformers ≥ 4.40`,
`scikit-learn ≥ 1.3`, `scipy ≥ 1.11`, `pandas`, `numpy`, `pyyaml`

GPU with ≥ 16 GB VRAM recommended for pre-training; fine-tuning fits on 8 GB.

---

## Dataset Preparation

Download Speechocean762 from OpenSLR and unpack so the layout matches:

```
data/
└── SPEECHOCEAN762/
    ├── train/
    │   ├── wav.scp          # utt_id  /abs/path/to/audio.wav
    │   ├── text             # utt_id  TRANSCRIPT
    │   └── utt2spk          # utt_id  spk_id
    ├── test/
    │   ├── wav.scp
    │   ├── text
    │   └── utt2spk
    └── resource/
        └── scores.json      # {utt_id: {total, accuracy, fluency, ...}}
```

Update `data.train_dir`, `data.val_dir`, and `data.scores_path` in
`configs/pretrain.yaml` and `configs/finetune.yaml` if your paths differ.

---

## Stage 1 — Self-supervised Pre-training

### Step 1a: Prepare k-means targets (iteration 1 — MFCC features)

```bash
cd multi_res_hubert
python pretrain.py --config configs/pretrain.yaml --prepare-kmeans
```

This runs MiniBatchKMeans on MFCC features extracted at HuBERT's native
20 ms frame rate (hop = 320 samples) and saves:

```
data/kmeans/
  kmeans_iter1.pkl       # fitted MiniBatchKMeans (K=100)
  labels_iter1.pkl       # {utt_id: {"hi": int16 array, "lo": int16 array}}
```

### Step 1b: Pre-train

```bash
python pretrain.py --config configs/pretrain.yaml
# Resume a crashed run:
python pretrain.py --config configs/pretrain.yaml \
    --resume checkpoints/pretrain/step_0010000.pt
```

Key hyperparameters (edit `configs/pretrain.yaml`):

| Parameter | Default | Notes |
|---|---|---|
| `training.warmup_steps` | 10 000 | Linear LR ramp |
| `training.learning_rate` | 1e-4 | Peak LR after warmup |
| `training.use_amp` | true | Mixed precision (requires CUDA) |
| `training.epochs` | 100 | — |
| `model.mask_prob` | 0.065 | Fraction of frames to mask |
| `model.mask_length` | 10 | Span length (frames) |
| `kmeans.n_clusters_hi` | 100 | Unit vocabulary size |

TensorBoard logs land in `logs/pretrain/` by default.

### Step 1c (Optional): Second k-means iteration on model features

```bash
# Set kmeans.iteration=2 and kmeans.model_checkpoint in pretrain.yaml, then:
python pretrain.py --config configs/pretrain.yaml --prepare-kmeans
python pretrain.py --config configs/pretrain.yaml
```

---

## Stage 2 — Fine-tuning on Speechocean762

```bash
# Fine-tune from HuggingFace pretrained weights (no pre-training checkpoint)
python train.py --config configs/finetune.yaml

# Warm-start from a pre-training checkpoint
python train.py --config configs/finetune.yaml \
    --pretrained checkpoints/pretrain/final_pretrain.pt

# Resume a stopped fine-tune run
python train.py --config configs/finetune.yaml \
    --resume checkpoints/finetune/best_model.pt
```

Key hyperparameters (edit `configs/finetune.yaml`):

| Parameter | Default | Notes |
|---|---|---|
| `training.warmup_steps` | 500 | Shorter warmup for fine-tuning |
| `training.learning_rate` | 1e-4 | — |
| `training.epochs` | 30 | — |
| `training.score_weights` | [1,1,1,1,2] | Upweights `total` dimension |
| `model.freeze_f1_layers` | true | Freeze bottom 4 transformer layers |

---

## Evaluation

```bash
python evaluate.py \
    --config     configs/finetune.yaml \
    --checkpoint checkpoints/finetune/best_model.pt \
    --output-dir results/eval/

# Also run hi-res vs lo-res ablation study
python evaluate.py ... --ablation

# Add scatter plots (requires matplotlib)
python evaluate.py ... --ablation --plot
```

**Console output example:**

```
Main Results (scores in [0, 10])
──────────────────────────────────────────────────────────────
 Dimension    │  PCC ↑  │  MSE ↓  │ RMSE ↓
──────────────────────────────────────────────────────────────
 accuracy     │ +0.7312 │  0.0148 │  0.1217
 fluency      │ +0.7891 │  0.0119 │  0.1091
 completeness │ +0.7104 │  0.0187 │  0.1368
 prosodic     │ +0.7425 │  0.0155 │  0.1245
 total        │ +0.8134 │  0.0097 │  0.0985
 ── avg ──    │ +0.7573 │  0.0141 │  0.1181
──────────────────────────────────────────────────────────────

Head Contribution Ablation
──────────────────────────────────────────────────────────────────────
 Head              │ PCC(total) │ MSE(total) │ PCC(avg) │ MSE(avg)
──────────────────────────────────────────────────────────────────────
 Combined (H₃+H₂) │   +0.8134 │     0.0097 │  +0.7573 │   0.0141
 Hi-res only (H₃) │   +0.7968 │     0.0108 │  +0.7401 │   0.0158
 Lo-res only (H₂) │   +0.7612 │     0.0132 │  +0.7089 │   0.0185
──────────────────────────────────────────────────────────────────────
```

Results are also written to `results/eval/metrics.csv` and
`results/eval/predictions.csv`.

---

## Expected Results

The table below shows approximate PCC on the Speechocean762 **test** split.
Exact numbers vary with GPU seed, number of pre-training epochs, and
whether a pre-training checkpoint is used.

| System | acc | flu | com | pro | total |
|---|---|---|---|---|---|
| HuBERT-base (fine-tune only) | 0.71 | 0.76 | 0.69 | 0.72 | 0.79 |
| **MultiRes-HuBERT (fine-tune only)** | **0.73** | **0.79** | **0.71** | **0.74** | **0.81** |
| MultiRes-HuBERT (pretrain → finetune) | ~0.75 | ~0.81 | ~0.73 | ~0.76 | ~0.84 |

> Numbers marked `~` are estimates; actual results depend on pre-training
> corpus size and number of iterations.

---

## Feature Extraction (for probing / analysis)

```bash
# Extract all four layer representations for the test set
python extract_features.py \
    --config      configs/finetune.yaml \
    --checkpoint  checkpoints/finetune/best_model.pt \
    --wav-scp     data/SPEECHOCEAN762/test/wav.scp \
    --output-dir  features/test/ \
    --layers      h0,h1,h2,h3

# Extract only f3 for a single file
python extract_features.py \
    --config     configs/finetune.yaml \
    --checkpoint checkpoints/finetune/best_model.pt \
    --audio      path/to/audio.wav \
    --output-dir features/single/ \
    --layers     h3
```

Output: one `.npy` file per layer per utterance.

---

## K-means Clustering (standalone)

Fit k-means on pre-extracted features for K ∈ {100, 200, 500}:

```bash
python kmeans_clustering.py \
    --features-dir features/train/ \
    --layer        h1 \
    --k            100,200,500 \
    --output-dir   data/kmeans/ \
    --save-labels \
    --plot
```

Outputs:
- `centroids_k100_h1.npy` — (K, H) centroid matrix (float32)
- `kmeans_k100_h1.pkl`    — serialised `MiniBatchKMeans` object
- `labels_k100_h1.pkl`    — `{utt_id: np.int16 array}` (pre-training compatible)
- `kmeans_analysis.png`   — elbow curve + Gini coefficient + size distribution

---

## Project Structure

```
multi_res_hubert/
│
├── train.py                   Fine-tuning entry point
├── pretrain.py                Pre-training entry point
├── evaluate.py                Evaluation + ablation study
├── extract_features.py        Extract h0/h1/h2/h3 representations
├── kmeans_clustering.py       Standalone k-means on extracted features
│
├── configs/
│   ├── pretrain.yaml          Pre-training hyperparameters
│   ├── finetune.yaml          Fine-tuning hyperparameters
│   └── base.yaml              Minimal reference config
│
├── data/
│   ├── dataset.py             Speechocean762Dataset (fine-tuning)
│   ├── pretrain_dataset.py    PretrainDataset + pretrain_collate_fn
│   └── collator.py            collate_fn for fine-tuning
│
├── model/
│   └── multi_res_hubert.py    Full architecture:
│                              DownsamplingModule, UpsamplingModule,
│                              ConvFeatureMasking, UnitPredictionHead,
│                              PronunciationScoreHead, MultiResHuBERT
│
└── training/
    ├── losses.py              PronunciationLoss, MultiResPretrainingLoss
    ├── trainer.py             Fine-tuning trainer (AMP + step scheduler)
    ├── pretrain_trainer.py    Pre-training trainer (AMP + masking + logging)
    ├── scheduler.py           Linear warmup + cosine decay
    ├── logger.py              WandB / TensorBoard abstraction
    └── kmeans.py              On-the-fly MFCC/model k-means for pre-training
```

---

## Citation

If you use this code, please cite the original Multi-resolution HuBERT paper
and the Speechocean762 dataset:

```bibtex
@inproceedings{multireshubert,
  title     = {Multi-Resolution {HuBERT}: Multi-Resolution Speech
               Self-Supervised Learning with Masked Unit Prediction},
  booktitle = {ICLR},
  year      = {2024},
}

@inproceedings{speechocean762,
  title     = {Speechocean762: An Open-Source Non-Native English Speech
               Corpus for Pronunciation Assessment},
  author    = {Zhang, Su and Gong, Yuan and Glass, James},
  booktitle = {Interspeech},
  year      = {2021},
}
```
