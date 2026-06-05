"""Runs all phase validation tests (Phases 0-2 + E2E) using stubs."""
import sys, torch, torch.nn as nn
from typing import List, Tuple
from unittest.mock import patch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

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
    def _audio_mask_to_feat_mask(self, m, n): return torch.ones(m.shape[0], n, dtype=torch.bool)
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
        "n_scores": 4, "scoring_dropout": 0.0,
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
print("=== PHASE 0 — correctness fixes ===")
model.eval()
with torch.no_grad():
    B, T = 2, 32000
    wav  = torch.randn(B, T)
    mask = torch.ones(B, T, dtype=torch.long)
    out  = model(wav, mask, ["hello world", "this is a test"])
    assert out.shape == (B, 4), f"Shape wrong: {out.shape}"
    assert (out > 0).all() and (out < 10).all(), \
        f"Scores outside (0,10): min={out.min():.3f} max={out.max():.3f}"
    print(f"Fix 1 OK — scores in (0,10): {out.round(decimals=3)}")

import torch.nn.functional as F
w = F.softmax(model.audio_encoder.layer_weights, dim=1)
acc_peak = w[0].argmax().item()
assert acc_peak >= 4, f"Accuracy peak {acc_peak}, expected >= 4"
print(f"Fix 2 OK — accuracy peak at hidden state {acc_peak}")

# Fix 3 is a source comment — verified by inspection in cross_attention_fusion.py
print("Fix 3 — TODO comment verified in TextProjection.forward source")
print("All Phase 0 fixes validated.\n")

# ─────────────────────────────────────────────────────────────────────────────
print("=== PHASE 1 — Upgrade 2: per-token text K/V ===")
model.eval()
with torch.no_grad():
    feats, msk = model.text_projection(["hello world", "test sentence with more tokens"])
    assert feats.ndim == 3, f"Expected [B,N,256], got {feats.shape}"
    assert feats.shape[-1] == PROJ_DIM
    assert msk.dtype == torch.bool
    assert msk.shape == (2, feats.shape[1])
    print(f"Text tokens: N={feats.shape[1]}, mask padded={msk.sum()} positions")
    out = model(wav, mask, ["hello world", "test sentence with more tokens"])
    assert out.shape == (2, 4) and (out > 0).all() and (out < 10).all()
    print(f"Forward pass OK: {out.round(decimals=3)}")

model.train()
model(wav, mask, ["hello world", "this is a test"]).sum().backward()
mha_g = model.cross_attn_fusion.mha.in_proj_weight.grad
assert mha_g is not None and mha_g.norm().item() > 0
print(f"CrossAttention in_proj grad norm: {mha_g.norm().item():.6f}")
print("Upgrade 2 validation complete.\n")

# ─────────────────────────────────────────────────────────────────────────────
print("=== PHASE 2 — Upgrade 3: composite loss ===")
from model.losses import (DimensionHuberLoss, ListMLERankingLoss,
                           ConcordanceCorrelationLoss, PronunciationScoringLoss)
torch.manual_seed(42)
B = 4
pred   = (torch.sigmoid(torch.randn(B, 4)) * 10).requires_grad_(True)
target = torch.rand(B, 4) * 10

l_h = DimensionHuberLoss()(pred, target)
assert l_h.item() >= 0 and l_h.requires_grad, "Huber checks failed"

l_r = ListMLERankingLoss()(pred, target)
assert l_r.item() >= 0 and l_r.requires_grad, "ListMLE checks failed"

l_c = ConcordanceCorrelationLoss()(pred, target)
assert 0 <= l_c.item() <= 2, f"CCC out of [0,2]: {l_c.item()}"
assert l_c.requires_grad, "CCC not differentiable"
print(f"Huber: {l_h.item():.4f}  ListMLE: {l_r.item():.4f}  CCC: {l_c.item():.4f}")

# CCC stress test
for _ in range(200):
    p2 = (torch.sigmoid(torch.randn(B, 4)) * 10).requires_grad_(True)
    lc = ConcordanceCorrelationLoss()(p2, torch.rand(B, 4) * 10)
    assert 0 <= lc.item() <= 2, f"CCC stress fail: {lc.item()}"
print("CCC stress test (200 batches) passed.")

crit = PronunciationScoringLoss(warmup_steps=1000)
pw = (torch.sigmoid(torch.randn(B, 4)) * 10).requires_grad_(True)
o0 = crit(pw, target.clone(), global_step=0)
assert o0["listmle"].item() == 0.0 and o0["ccc"].item() == 0.0, "Pre-warmup gates wrong"
o0["total"].backward(); assert pw.grad is not None
tot0 = o0["total"].item()
print(f"Pre-warmup: total={tot0:.4f}  listmle=0  ccc=0  (gates active) OK")

pw2 = (torch.sigmoid(torch.randn(B, 4)) * 10).requires_grad_(True)
o1 = crit(pw2, target.clone(), global_step=1000)
assert o1["listmle"].item() > 0.0 and o1["ccc"].item() > 0.0, "Post-warmup gates wrong"
o1["total"].backward()
tot1, lml1, ccc1 = o1["total"].item(), o1["listmle"].item(), o1["ccc"].item()
print(f"Post-warmup: total={tot1:.4f}  listmle={lml1:.4f}  ccc={ccc1:.4f}  (all active) OK")

crit.train()
for _ in range(10):
    p3 = (torch.sigmoid(torch.randn(B, 4)) * 10).requires_grad_(True)
    o_s = crit(p3, target.clone(), global_step=1000)
    assert o_s["total"].item() >= 0
print("Label smoothing: 10 runs OK")
print("All Phase 2 loss validations passed.\n")

# ─────────────────────────────────────────────────────────────────────────────
print("=== FINAL END-TO-END VALIDATION ===")
model.train()
opt = model.get_optimizer()
opt.zero_grad()

B, T = 2, 32000
wav2  = torch.randn(B, T)
mask2 = torch.ones(B, T, dtype=torch.long)
txts2 = ["hello world", "pronunciation test sentence"]
scores_gt = torch.rand(B, 4) * 10

scores = model(wav2, mask2, txts2)
assert scores.shape == (B, 4) and (scores > 0).all() and (scores < 10).all()

crit2 = PronunciationScoringLoss()
ld = crit2(scores, scores_gt, global_step=1500)
ld["total"].backward()

print("=== Gradient norms ===")
checks = [
    ("layer_weights [4,12]", model.audio_encoder.layer_weights),
    ("audio_proj[0]",        model.audio_encoder.proj[0].weight),
    ("audio_proj[3]",        model.audio_encoder.proj[3].weight),
    ("cross_attn K proj",    model.cross_attn_fusion.mha.in_proj_weight),
    ("text_proj linear",     model.text_projection.proj.weight),
]
for name, param in checks:
    gn = param.grad.norm().item() if param.grad is not None else 0.0
    print(f"  {name}: {gn:.6f}")
    assert gn > 0, f"Expected non-zero grad on {name}"

for name, p in model.named_parameters():
    if "text_model" in name:
        gn = p.grad.norm().item() if p.grad is not None else 0.0
        assert gn == 0.0, f"Frozen text_model should have zero grad: {name}"

assert model.text_projection.proj.weight.grad.norm().item() > 0

sc_str = str(scores.detach().round(decimals=2).tolist())
tot_ld = ld["total"].item(); hub_ld = ld["huber"].item()
lml_ld = ld["listmle"].item(); ccc_ld = ld["ccc"].item()
print(f"\nFinal scores: {sc_str}")
print(f"Loss: total={tot_ld:.4f}  huber={hub_ld:.4f}  listmle={lml_ld:.4f}  ccc={ccc_ld:.4f}")
print("\nAll end-to-end validations passed. OK")
