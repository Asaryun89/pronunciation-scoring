# Task: Implement BGE Text Embedder for Pronunciation Scoring Model

## Context

You are working on a pronunciation scoring system. The team is building a model that:
- Takes audio (user reading a script) + the script text as input
- Outputs 5 pronunciation scores: accuracy, completeness, fluency, prosody, total

The original paper used Faster-Whisper ASR + Qwen3-Embedding-0.6B for the text branch. The goal is to replace this with a lighter, cleaner text embedding using `BAAI/bge-small-en-v1.5`.

**Architecture overview:**
```
Audio → HuBERT → audio features (B, T, 256)
                                          ↓
Script text → BGE → mean pool → (B, 1, 256) ← text feature
                                          ↓
                    CrossAttentionFusion(Q=audio, K/V=text)
                                          ↓
                    Transformer Encoder (2 layers)
                                          ↓
                    Mean pool → MLP → 5 scores
```

Note: text branch outputs **(B, 1, 256)** — one vector per sentence (not per token). This matches the original paper design (mask-aware mean pool).

## Existing files (DO NOT modify these)

```
models/scoring_heads.py     # CrossAttentionFusion, MLPScoringHead — keep as-is
models/phoneme_vocab.py     # ARPABET vocab — keep as-is
models/phoneme_embedder.py  # PhonemeEmbedder — keep as-is (for ablation later)
```

## Files to create

### File 1: `models/text_embedder.py`

Class `BGETextEmbedder(nn.Module)` that wraps `BAAI/bge-small-en-v1.5`.

Constructor args:
- `d_model: int = 256`
- `model_name: str = "BAAI/bge-small-en-v1.5"`
- `freeze_encoder: bool = True` — freeze BGE weights by default (fine-tune only proj layer)
- `dropout: float = 0.1`

BGE-small hidden size = 384. Need to project to d_model.

Components:
- `self.encoder = AutoModel.from_pretrained(model_name)` 
- If `freeze_encoder=True`: freeze all encoder params
- `self.proj = nn.Linear(384, d_model)`
- `self.norm = nn.LayerNorm(d_model)`
- `self.dropout = nn.Dropout(dropout)`

Static method `_mean_pool(last_hidden_state, attention_mask) -> Tensor`:
```python
# Masked mean pooling — ignore padding tokens
# last_hidden_state: (B, L, H)
# attention_mask: (B, L)  1=valid, 0=pad
# Returns: (B, H)
token_embeddings = last_hidden_state
input_mask_expanded = attention_mask.unsqueeze(-1).float()
return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)
```

Forward signature:
```python
def forward(
    self,
    input_ids: torch.Tensor,        # (B, L) tokenized script
    attention_mask: torch.Tensor,   # (B, L) 1=valid, 0=pad
) -> torch.Tensor:                  # (B, 1, d_model)
```

Forward steps:
1. Run BGE encoder: `out = self.encoder(input_ids, attention_mask)`
2. Mean pool: `text_emb = self._mean_pool(out.last_hidden_state, attention_mask)` → `(B, 384)`
3. Project: `text_emb = self.proj(text_emb)` → `(B, d_model)`
4. Normalize + dropout: `text_emb = self.dropout(self.norm(text_emb))`
5. Unsqueeze: `return text_emb.unsqueeze(1)` → `(B, 1, d_model)`

The unsqueeze at step 5 is important — CrossAttentionFusion expects `(B, L, d_model)` for K/V. We use L=1 (one vector per sentence).

Add full docstrings explaining:
- Why mean pooling (not CLS token) — BGE is designed for mean pooling
- Why L=1 in output — matching original paper design (sentence-level conditioning)
- Why freeze_encoder=True by default — BGE already has strong sentence representations; fine-tuning risks catastrophic forgetting on small SpeechOcean dataset

### File 2: `models/scoring_model.py` (NEW — replaces old version if exists)

Class `HubertScoringModel(nn.Module)` — same as before but text branch uses `BGETextEmbedder`.

Constructor args:
```python
def __init__(
    self,
    hubert_model_name: str = "facebook/hubert-base-ls960",
    bge_model_name: str = "BAAI/bge-small-en-v1.5",
    d_model: int = 256,
    num_heads: int = 8,
    num_audio_transformer_layers: int = 1,
    num_fusion_transformer_layers: int = 2,
    mlp_hidden_layers: int = 2,
    dropout: float = 0.1,
    freeze_feature_extractor: bool = True,
    num_unfreeze_hubert_layers: int = 12,
    freeze_bge: bool = True,
    num_aspects: int = 5,
    prosody_feat_dim: int = 5,
)
```

Components (same as before, only text branch changes):
- HuBERT backbone with layer-weighted sum (same as before)
- `self.audio_proj = nn.Linear(hubert_hidden, d_model)`
- `self.text_embedder = BGETextEmbedder(d_model, bge_model_name, freeze_encoder=freeze_bge, dropout=dropout)`
- Optional audio transformer
- `self.cross_attn = CrossAttentionFusion(d_model, num_heads, dropout)`
- Fusion transformer (2 layers)
- `self.scoring_head = MLPScoringHead(...)`
- `self.prosody_head = nn.Sequential(...)`

**DO NOT include PhonemeEmbedder in this file.** Keep it separate.

Forward signature:
```python
def forward(
    self,
    input_values: torch.Tensor,         # (B, num_samples) raw audio
    text_input_ids: torch.Tensor,       # (B, L) tokenized script
    text_attention_mask: torch.Tensor,  # (B, L) 1=valid 0=pad
    audio_attention_mask: Optional[torch.Tensor] = None,
    **_,
) -> Dict[str, torch.Tensor]:
```

Forward steps:
1. Audio path: HuBERT → layer weighted sum → `audio_proj` → optional audio transformer → `audio_emb (B, T, 256)`
2. Text path: `self.text_embedder(text_input_ids, text_attention_mask)` → `text_emb (B, 1, 256)`
3. Fusion: `cross_attn(audio_emb, text_emb)` — no key_padding_mask needed (L=1, no padding)
4. `fusion_transformer(fused)` → mean pool → `utt_emb (B, 256)`
5. Heads: `scoring_head(utt_emb)` and `prosody_head(utt_emb)`

Returns:
```python
{
    "sent_pred": (B, 5),      # in [0, 1]
    "prosody_pred": (B, 5),
}
```

All transformer layers: `norm_first=True, batch_first=True, enable_nested_tensor=False, dim_feedforward=4*d_model`.

### File 3: `utils/dataset.py` (NEW)

`SpeechOcean762Dataset` — same data loading as before, but tokenizes the script text for BGE instead of converting to phonemes.

Module-level constants:
```python
SCORE_KEYS = ["accuracy", "completeness", "fluency", "prosody", "total"]
SCORE_MAX = 10.0
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_MAX_LENGTH = 128  # scripts are short
```

Class `SpeechOcean762Dataset(Dataset)`:

Constructor:
```python
def __init__(
    self,
    split: str = "train",
    max_audio_seconds: float = 20.0,
    bge_model_name: str = BGE_MODEL_NAME,
    max_text_length: int = BGE_MAX_LENGTH,
)
```

- Load dataset: `load_dataset("mispeech/speechocean762", split=split)`
- Load tokenizer: `AutoTokenizer.from_pretrained(bge_model_name)`
- Store tokenizer for use in `__getitem__`

`__getitem__` returns:
```python
{
    "input_values": tensor(audio, float32),   # (T,)
    "text_input_ids": tensor (L,) long,       # tokenized script
    "text_attention_mask": tensor (L,) long,  # 1=valid
    "prosody_feats": tensor (5,) float,       # RMS, ZCR, peak, pitch_mean, pitch_std
    "sent_scores": tensor (5,) float,         # normalized to [0, 1]
}
```

Text tokenization:
```python
encoded = self.tokenizer(
    sample["text"],
    max_length=self.max_text_length,
    padding=False,        # padding done in collate_fn
    truncation=True,
    return_tensors="pt",
)
text_input_ids = encoded["input_ids"].squeeze(0)         # (L,)
text_attention_mask = encoded["attention_mask"].squeeze(0)  # (L,)
```

`prosody_feats` extraction: same as before (RMS, ZCR, peak_rate, pitch_mean, pitch_std via librosa). Wrap in try/except, fallback to zeros.

`normalize_scores`: same as before (divide by 10.0).

`collate_fn(batch)` function:
- Pad `input_values` with zeros → emit `audio_attention_mask` (1=valid, 0=pad) as long tensor
- Pad `text_input_ids` with tokenizer.pad_token_id → emit `text_attention_mask` (1=valid, 0=pad) as long tensor
- Stack `prosody_feats`, `sent_scores`

Returns:
```python
{
    "input_values": (B, T_max),
    "audio_attention_mask": (B, T_max),   # long
    "text_input_ids": (B, L_max),         # long
    "text_attention_mask": (B, L_max),    # long
    "prosody_feats": (B, 5),
    "sent_scores": (B, 5),
}
```

### File 4: `models/train.py` (NEW)

Training script — same structure as before, but uses text inputs instead of phoneme inputs.

`TrainConfig` dataclass:
```python
@dataclass
class TrainConfig:
    # Model
    hubert_model_name: str = "facebook/hubert-base-ls960"
    bge_model_name: str = "BAAI/bge-small-en-v1.5"
    d_model: int = 256
    num_heads: int = 8
    num_audio_transformer_layers: int = 1
    num_fusion_transformer_layers: int = 2
    mlp_hidden_layers: int = 2
    dropout: float = 0.1
    num_unfreeze_hubert_layers: int = 12
    freeze_bge: bool = True

    # Optimization
    epochs: int = 10
    batch_size: int = 4
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0

    # Loss weights
    lambda_sent: float = 1.0
    lambda_prosody: float = 0.5

    # Data
    max_audio_seconds: float = 20.0
    max_text_length: int = 128
    num_workers: int = 2

    # Runtime
    output_dir: str = "runs/bge_exp1"
    seed: int = 42
    log_every: int = 50
    fp16: bool = True
```

CLI args: `--output-dir, --epochs, --batch-size, --lr, --weight-decay, --num-workers, --seed, --no-fp16, --hubert-model-name, --bge-model-name, --no-freeze-bge`

`ScoringLoss` — same as before:
- Smooth L1 for sent_pred vs sent_scores
- Smooth L1 for prosody_pred vs standardized prosody_feats
- `total = lambda_sent * sent_loss + lambda_prosody * prosody_loss`

`train_one_epoch` — same structure, but batch keys are now:
```python
outputs = model(
    input_values=batch["input_values"],
    text_input_ids=batch["text_input_ids"],
    text_attention_mask=batch["text_attention_mask"],
    audio_attention_mask=batch["audio_attention_mask"],
)
```

`evaluate` — same structure.

`compute_correlations` — same (Pearson, Spearman, MAE per SCORE_KEYS).

`main()`:
1. Parse args → TrainConfig
2. Set seed, create output_dir, save config.json
3. Load train/test datasets with `SpeechOcean762Dataset`
4. Create DataLoaders with `collate_fn`
5. Build `HubertScoringModel`
6. Print param counts — include breakdown:
   ```
   HuBERT params:     90.1M (trainable: X.XM)
   BGE params:        22.7M (trainable: 0 if frozen)
   Other params:      X.XM
   Total trainable:   X.XM
   ```
7. AdamW on `requires_grad=True` params only
8. Linear warmup + linear decay scheduler
9. AMP GradScaler
10. Train loop: epoch → train_one_epoch → evaluate → log CSV → save best checkpoint
11. Best checkpoint criterion: highest Pearson on `total` score

## General code requirements

- `from __future__ import annotations` at top of every file
- Full type hints on all functions/methods
- Docstrings: module-level + class-level + all public methods
- `from transformers import AutoModel, AutoTokenizer` (not model-specific imports)
- No `sys.path` hacks except in `train.py` for script execution
- All constants in UPPER_CASE
- librosa imported lazily inside prosody function

## After creating files

1. Create `models/__init__.py` and `utils/__init__.py` if missing (empty files)

2. Syntax check all 4 files:
```bash
python -c "
import ast
files = [
    'models/text_embedder.py',
    'models/scoring_model.py',
    'utils/dataset.py',
    'models/train.py',
]
for f in files:
    ast.parse(open(f).read())
    print(f'OK: {f}')
"
```

3. Import check (without loading weights):
```bash
python -c "from models.text_embedder import BGETextEmbedder; print('OK: BGETextEmbedder')"
python -c "from models.scoring_model import HubertScoringModel; print('OK: HubertScoringModel')"
python -c "from utils.dataset import SpeechOcean762Dataset, collate_fn; print('OK: dataset')"
python -c "from models.train import TrainConfig; print('OK: TrainConfig')"
```

4. Check BGE is installable:
```bash
pip show transformers | grep Version
# Should be >= 4.30.0 for BGE support
```

5. Print model structure sanity check:
```bash
python -c "
from models.scoring_model import HubertScoringModel
import torch
# Init without loading weights for quick check
print('Model structure check...')
# Just verify the class can be imported and instantiated description is correct
print('OK')
"
```

## DO NOT touch these files
- `models/phoneme_vocab.py`
- `models/phoneme_embedder.py`
- `models/scoring_heads.py`
- `models/asr_aligner.py`
- `models/ctc_aligner.py`
- `inference/*`
- `notebook*/*`
- Any test files

## Acceptance criteria

- All 4 files compile (no SyntaxError)
- Imports resolve
- `BGETextEmbedder` outputs `(B, 1, d_model)` — NOT `(B, L, d_model)`
- `HubertScoringModel` forward accepts `text_input_ids` and `text_attention_mask` (NOT `phoneme_ids`)
- `collate_fn` produces correct keys: `input_values, audio_attention_mask, text_input_ids, text_attention_mask, prosody_feats, sent_scores`
- No imports of: `phoneme_vocab`, `PhonemeEmbedder`, `g2p`, `faster_whisper`, `Qwen`
- `--help` on train.py shows all CLI flags
- `freeze_bge=True` by default — BGE params have `requires_grad=False`

## Notes

- BGE-small hidden dim = **384** (not 768 like BERT-base, not 1024 like large models). Double check this when writing `nn.Linear(384, d_model)`.
- BGE uses mean pooling, NOT CLS token pooling. This is by design (stated in BGE paper/readme).
- SpeechOcean `sample["text"]` field contains the script — use this directly for tokenization.
- `collate_fn` should use `tokenizer.pad_token_id` for text padding, NOT 0 (which may conflict with actual token IDs).

## TODO comments to include in code

Add these comments in `scoring_model.py` for future reference:

```python
# TODO(phase2): Swap HuBERT backbone to HuBERT-CTC from Linh's checkpoint
# See: branch linh/hubert_only, model: hubert-large-speechocean-ctc
# Requires: update audio_proj dim 768 → 1024, layer_weights size 13 → 25

# TODO(ablation): Compare BGETextEmbedder vs PhonemeEmbedder
# PhonemeEmbedder is in models/phoneme_embedder.py
# Hypothesis: phoneme gives better cross-attention alignment (sequence vs 1 vector)
# Run: python -m models.train --output-dir runs/bge_exp
#      python -m models.train_phoneme --output-dir runs/phoneme_exp (Phase 3)
```
