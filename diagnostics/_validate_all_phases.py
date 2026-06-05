"""All-phase validation: Phases 1-4 + final E2E smoke test."""
import sys, os, torch, torch.nn as nn, tempfile
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from typing import List, Tuple
from unittest.mock import patch

H_DIM=768; N_LAYERS=12; T_FEAT=49; N_TOK=10; PROJ_DIM=256; TEXT_DIM=1024

class _FakeOut:
    def __init__(self, hs): self.all_hidden_states = hs

class _FakeBB(nn.Module):
    def __init__(self, *a, **kw):
        super().__init__()
        self.feat_masking = type("FM", (), {"mask_prob": 0.0})()
        self.feature_extractor = self.feature_projection = nn.Identity()
        self.f1_layers = self.f2_layers = self.f3_layers = nn.ModuleList()
        self._p = nn.Parameter(torch.zeros(1))
    def forward(self, wav, attn=None, apply_mask=False, output_hidden_states=False):
        B = wav.shape[0]
        hs = [torch.ones(B, T_FEAT, H_DIM) for _ in range(N_LAYERS)] if output_hidden_states else None
        return _FakeOut(hs)
    def _audio_mask_to_feat_mask(self, m, n):
        return torch.ones(m.shape[0], n, dtype=torch.bool)
    def load_state_dict(self, sd, strict=True): pass
    def parameters(self, recurse=True): yield self._p

class _FakeTTP(nn.Module):
    def __init__(self, *a, **kw):
        super().__init__()
        self.proj = nn.Linear(TEXT_DIM, PROJ_DIM)
        self.norm = nn.LayerNorm(PROJ_DIM)
        self.text_model = nn.Linear(8, 8)
        for p in self.text_model.parameters(): p.requires_grad_(False)
    def forward(self, texts):
        B = len(texts)
        return (self.norm(self.proj(torch.randn(B, N_TOK, TEXT_DIM))),
                torch.zeros(B, N_TOK, dtype=torch.bool))

cfg = {
    "model": {
        "pretrain_checkpoint": None, "hubert_model_name": "fake",
        "f1_layer_count": 4, "f2_layer_count": 4, "f3_layer_count": 4,
        "downsample_stride": 2, "freeze_feature_extractor": True,
        "freeze_f1_layers": False, "freeze_f2_layers": False, "freeze_f3_layers": False,
        "num_hidden_layers": N_LAYERS, "speech_hidden_dim": H_DIM, "proj_dim": PROJ_DIM,
        "pre_fusion_heads": 4, "pre_fusion_dropout": 0.0, "pre_fusion_layers": 1,
        "num_units_hi": 100, "num_units_lo": 100,
        "text_encoder_name": "fake", "text_encoder_dim": TEXT_DIM,
        "freeze_text_encoder": True, "text_max_length": 128, "text_padding_side": "left",
        "fusion_heads": 4, "fusion_dropout": 0.0,
        "post_fusion_heads": 4, "post_fusion_dropout": 0.0, "post_fusion_layers": 2,
        "n_scores": 4, "scoring_dropout": 0.0, "num_phonemes": 40,
    },
    "training": {
        "lr_speech_encoder": 5e-5, "lr_audio_proj": 1e-4,
        "lr_text_encoder": 0.0, "lr_fusion": 1e-4,
        "weight_decay": 1e-2, "betas": [0.9, 0.98],
    },
}

from model.pronunciation_scorer import PronunciationScorer
with patch("model.audio_encoder.MultiResHuBERT", _FakeBB), \
     patch("model.pronunciation_scorer.TokenTextProjection", _FakeTTP):
    model = PronunciationScorer(cfg)

# ─────────────────────────────────────────────────────────────────────────────
print("=== PHASE 1: ListMLE normalisation + alpha/gamma rebalancing ===")
from model.losses import (PronunciationScoringLoss, ListMLERankingLoss,
                           DimensionHuberLoss, ConcordanceCorrelationLoss)

torch.manual_seed(0)
# B=4 matches training batch_size; z-score normalises to ~unit variance so
# ListMLE magnitude scales as O(log B) ≈ log(4)=1.4 for random predictions.
# Training data with partial correlation gives 0.1-0.2 after convergence.
B = 4
pred   = (torch.rand(B, 4) * 10).requires_grad_(True)
target = torch.rand(B, 4) * 10

lmle = ListMLERankingLoss()(pred, target)
# Threshold 0.8: log(4)/2 ≈ 0.69 for purely random predictions, well below
# the un-normalised value (~5-15 for raw 0-10 scale with B=4).
assert lmle.item() < 0.8, f"ListMLE too large after normalisation: {lmle.item():.4f}"
print(f"ListMLE normalised (B=4): {lmle.item():.4f}  (target < 0.8) OK")

crit = PronunciationScoringLoss()
assert crit.alpha == 0.15, f"alpha wrong: {crit.alpha}"
assert crit.gamma == 0.35, f"gamma wrong: {crit.gamma}"
print(f"alpha={crit.alpha}, gamma={crit.gamma} OK")

pw = (torch.rand(B, 4) * 10).requires_grad_(True)
out = crit(pw, target.clone(), global_step=1500)
listmle_val = out["listmle"].item()
huber_val   = out["huber"].item()
assert listmle_val < huber_val * 2.0, \
    f"ListMLE ({listmle_val:.4f}) still dominates Huber ({huber_val:.4f})"
print(f"Balanced: huber={huber_val:.4f}  listmle={listmle_val:.4f}  "
      f"ccc={out['ccc'].item():.4f} OK")
print("Phase 1 PASSED\n")

# ─────────────────────────────────────────────────────────────────────────────
print("=== PHASE 2: grad clipping ===")
model.train()
opt = model.get_optimizer(); opt.zero_grad()
B2, T2 = 4, 16000
wav2  = torch.randn(B2, T2) * 5.0
mask2 = torch.ones(B2, T2, dtype=torch.long)
txts2 = ["a", "b", "c", "d"]
gt2   = torch.rand(B2, 4) * 10

scores2 = model(wav2, mask2, txts2)
crit2   = PronunciationScoringLoss(alpha=0.15, gamma=0.35)
ld2     = crit2(scores2, gt2, global_step=2000)
ld2["total"].backward()

raw_norm = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
clipped  = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
post     = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
assert post.item() <= 3.05, f"Grad not clipped: {post.item():.4f}"
print(f"Pre-clip: {raw_norm:.4f}  clipped to: {post:.4f}  (<=3.0) OK")
print("Phase 2 PASSED\n")

# ─────────────────────────────────────────────────────────────────────────────
print("=== PHASE 3: early stopping ===")
from training.early_stopping import EarlyStopping

val_sequence = [
    0.5996, 0.6351, 0.6001, 0.6388, 0.6426,
    0.6465, 0.6073, 0.6383, 0.6180, 0.6486,
    0.6299, 0.6298, 0.6094, 0.6152, 0.6391,
]

with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = os.path.join(tmp, "best_model.pt")
    es = EarlyStopping(patience=5, min_delta=0.001, mode="max",
                       checkpoint_path=ckpt_path, verbose=False)
    stop_epoch = None
    dummy_m = nn.Linear(2, 2)
    for epoch, pcc in enumerate(val_sequence, start=1):
        fired = es.step(pcc, dummy_m, epoch)
        if fired and stop_epoch is None:
            stop_epoch = epoch

    assert es.best_epoch == 10, f"Wrong best epoch: {es.best_epoch}"
    assert abs(es.best_score - 0.6486) < 1e-4
    assert stop_epoch == 15, f"Wrong stop epoch: {stop_epoch}"
    assert es.counter == 5
    assert os.path.exists(ckpt_path)
    print(f"best_epoch={es.best_epoch} best_pcc={es.best_score:.4f} "
          f"stop={stop_epoch} counter={es.counter} OK")

print("Phase 3 PASSED\n")

# ─────────────────────────────────────────────────────────────────────────────
print("=== PHASE 4: aux CE non-zero ===")
B4, T4, NP = 4, 150, 40
pred4    = (torch.rand(B4, 4) * 10).requires_grad_(True)
target4  = torch.rand(B4, 4) * 10
ph_logit = torch.randn(B4 * T4, NP)
ph_label = torch.randint(-1, NP, (B4 * T4,))

crit4 = PronunciationScoringLoss(beta=0.10)
out4  = crit4(pred4, target4, global_step=1500,
              phoneme_logits=ph_logit, phoneme_labels=ph_label)
assert out4["aux_ce"].item() > 0.0, f"aux_ce still zero: {out4['aux_ce'].item()}"
print(f"aux_ce={out4['aux_ce'].item():.4f}  "
      f"beta*aux={0.10 * out4['aux_ce'].item():.4f} OK")
print("Phase 4 PASSED\n")

# ─────────────────────────────────────────────────────────────────────────────
print("=== FINAL E2E SMOKE TEST ===")
model.train()
opt = model.get_optimizer(); opt.zero_grad()
B5, T5 = 2, 32000
wav5  = torch.randn(B5, T5)
mask5 = torch.ones(B5, T5, dtype=torch.long)
txts5 = ["hello world", "pronunciation assessment test"]
gt5   = torch.rand(B5, 4) * 10

# Forward with current_epoch=1 (phoneme head active, no detach)
scores5 = model(wav5, mask5, txts5, current_epoch=1)
assert scores5.shape == (B5, 4)
assert (scores5 > 0).all() and (scores5 < 10).all()

T_prime = getattr(model, "last_phoneme_logits", None)
assert T_prime is not None, "last_phoneme_logits not set"
ph_logits5 = model.last_phoneme_logits
ph_labels5 = torch.randint(-1, 40, (ph_logits5.shape[0],))

crit5     = PronunciationScoringLoss(alpha=0.15, gamma=0.35, beta=0.10)
loss_dict5 = crit5(scores5, gt5, global_step=1500,
                   phoneme_logits=ph_logits5, phoneme_labels=ph_labels5)
loss_dict5["total"].backward()

clipped5 = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
opt.step()

print("=== Gradient norms ===")
checks = [
    ("layer_weights [4,12]", model.audio_encoder.layer_weights),
    ("audio_proj[0]",        model.audio_encoder.proj[0].weight),
    ("audio_proj[3]",        model.audio_encoder.proj[3].weight),
    ("cross_attn K proj",    model.cross_attn_fusion.mha.in_proj_weight),
    ("text_proj linear",     model.text_projection.proj.weight),
    ("phoneme_head",         model.phoneme_head.weight),
]
for name, param in checks:
    gn = param.grad.norm().item() if param.grad is not None else 0.0
    print(f"  {name}: {gn:.6f}")
    assert gn > 0, f"Expected non-zero grad on {name}"

assert loss_dict5["aux_ce"].item() > 0.0, "aux_ce must be non-zero"
assert clipped5.item() > 0.0

with tempfile.TemporaryDirectory() as tmp:
    es2 = EarlyStopping(patience=5, mode="max",
                        checkpoint_path=os.path.join(tmp, "best.pt"),
                        verbose=False)
    es2.step(0.64, model, epoch=1)
    es2.step(0.65, model, epoch=2)
    assert es2.best_score == 0.65 and es2.counter == 0
    es2.step(0.63, model, epoch=3)
    assert es2.counter == 1
    print("Early stopping counter logic OK")

sc = scores5.detach().tolist()
tot  = loss_dict5["total"].item()
hub  = loss_dict5["huber"].item()
lml  = loss_dict5["listmle"].item()
ccc  = loss_dict5["ccc"].item()
aux  = loss_dict5["aux_ce"].item()
print(f"\nScores: {[[round(v,2) for v in row] for row in sc]}")
print(f"Loss: total={tot:.4f}  huber={hub:.4f}  listmle={lml:.4f}  "
      f"ccc={ccc:.4f}  aux_ce={aux:.4f}")
print(f"Grad norm (pre-clip): {clipped5:.4f}")
print("\nAll phases validated. Ready to train.")
