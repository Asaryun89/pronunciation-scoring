"""
visualize_architecture.py

Print a detailed ASCII diagram and save a matplotlib flowchart of the
PronunciationScorer architecture (FlatHuBERT + Qwen3 cross-attention).

Usage:
    python visualize_architecture.py                # ASCII + PNG diagram
    python visualize_architecture.py --no-plot      # ASCII only
    python visualize_architecture.py --summary      # + torchinfo param counts
    python visualize_architecture.py --out arch.png # custom output path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Force UTF-8 on Windows so box-drawing characters print correctly.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# ASCII diagram
# ─────────────────────────────────────────────────────────────────────────────

ASCII = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            PronunciationScorer — Full Architecture                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   AUDIO PATH                           TEXT PATH                            ║
║   ──────────────────────────           ──────────────────────────           ║
║   raw waveform  [B, T_audio]           transcripts  List[str]               ║
║          │                                    │                             ║
║   ┌──────▼──────────────────────┐     ┌───────▼──────────────────┐         ║
║   │  FlatHuBERT  (pretrained)   │     │  Qwen3-Embedding-0.6B    │         ║
║   │  facebook/hubert-large      │     │  (frozen, 596M params)   │         ║
║   │  ─────────────────────────  │     │  ─────────────────────   │         ║
║   │  CNN f₀  [B, T', 1024]      │     │  padding_side = "left"   │         ║
║   │     ↓  ConvFeatureMasking   │     │  instruct prefix          │         ║
║   │     ↓  pos_conv + LN + Drop │     │  mask-aware mean pool     │         ║
║   │                             │     │  L2 normalise             │         ║
║   │  TransformerEncoder ×24     │     └───────────┬──────────────┘         ║
║   │  (flat — single resolution) │                 │                         ║
║   │  no DOWN / UP blocks        │          [B, 1024]  float32               ║
║   │                             │                 │                         ║
║   │  output_hidden_states=True  │     ┌───────────▼──────────────┐         ║
║   │  24 × [B, T', 1024]         │     │  TextProjection          │         ║
║   └──────────┬──────────────────┘     │  Linear(1024→256)        │         ║
║              │                        │  + LayerNorm             │         ║
║   ┌──────────▼─────────────┐          │  unsqueeze dim=1         │         ║
║   │  Learnable Weighted Sum │          └───────────┬──────────────┘         ║
║   │  softmax(layer_weights) │                      │                         ║
║   │  Σ wᵢ · hiddenᵢ         │             [B, 1, 256]  (K / V)              ║
║   │  → [B, T', 1024]        │                      │                         ║
║   └──────────┬──────────────┘                      │                         ║
║              │                                      │                         ║
║   ┌──────────▼─────────────┐                        │                         ║
║   │  Linear(1024 → 256)    │                        │                         ║
║   └──────────┬─────────────┘                        │                         ║
║              │                                      │                         ║
║   ┌──────────▼─────────────┐                        │                         ║
║   │  TransformerEncoder ×1 │                        │                         ║
║   │  pre-LN, 4 heads       │                        │                         ║
║   └──────────┬─────────────┘                        │                         ║
║              │                                      │                         ║
║        [B, T', 256]  (Q) ──────────────────────────┘                         ║
║                                                                              ║
║                         FUSION                                               ║
║                         ──────────────────────────                           ║
║              ┌──────────────────────────────────┐                           ║
║              │  CrossAttentionFusion            │                           ║
║              │  MHA(Q=audio, K=text, V=text)    │                           ║
║              │  4 heads · residual · LayerNorm  │                           ║
║              └──────────────┬───────────────────┘                           ║
║                             │  [B, T', 256]                                  ║
║              ┌──────────────▼───────────────────┐                           ║
║              │  TransformerEncoder ×2           │                           ║
║              │  pre-LN, 4 heads                │                           ║
║              └──────────────┬───────────────────┘                           ║
║                             │  [B, T', 256]                                  ║
║              ┌──────────────▼───────────────────┐                           ║
║              │  Masked Mean Pool  (over T')     │                           ║
║              └──────────────┬───────────────────┘                           ║
║                             │  [B, 256]                                      ║
║              ┌──────────────▼───────────────────┐                           ║
║              │  MLPScoringHead                  │                           ║
║              │  5 × independent MLP heads       │                           ║
║              │  Linear(256,128)→ReLU→Dropout    │                           ║
║              │  →Linear(128,64)→ReLU            │                           ║
║              │  →Linear(64,1)→Sigmoid           │                           ║
║              └──────────────┬───────────────────┘                           ║
║                             │  [B, 5] ∈ (0,1)  × 10                         ║
║                             ▼                                                ║
║              ┌──────────────────────────────────┐                           ║
║              │  [total, accuracy, fluency,      │                           ║
║              │   prosodic]                      │                           ║
║              │  ∈ (0, 10)  MOS range            │                           ║
║              └──────────────────────────────────┘                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Optimizer param groups:                                                     ║
║    A  FlatHuBERT backbone (trainable layers)        lr = 5e-5               ║
║    B  audio proj + layer_weights + pre-transformer  lr = 1e-4               ║
║    C  text encoder (Qwen3, frozen)                  lr = 0.0                ║
║    D  fusion + post-transformer + scoring head      lr = 1e-4               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# Parameter count table
# ─────────────────────────────────────────────────────────────────────────────

def _param_table() -> None:
    rows = [
        ("Component",                          "Params",   "Trainable"),
        ("─" * 40,                             "─" * 10,  "─" * 10),
        ("FlatHuBERT  (hubert-large-ll60k)",   "~305 M",  "~300 M*"),
        ("  CNN feature extractor (f₀)",        "~5 M",    "frozen"),
        ("  feature projection",                "~1 M",    "frozen"),
        ("  pos_conv + LayerNorm",              "~0.7 M",  "yes"),
        ("  TransformerEncoder ×24  (flat)",    "~295 M",  "yes"),
        ("  UnitPredictionHead  (unused)",      "~0.1 M",  "frozen†"),
        ("Layer weights  (24 scalars)",         "< 0.1 K", "yes"),
        ("Linear proj  1024 → 256",             "0.26 M",  "yes"),
        ("Pre-fusion TransformerEncoder ×1",    "0.79 M",  "yes"),
        ("─" * 40,                             "─" * 10,  "─" * 10),
        ("Qwen3-Embedding-0.6B",               "596 M",   "frozen"),
        ("TextProjection  1024 → 256",         "0.26 M",  "yes"),
        ("CrossAttentionFusion  (MHA 4h)",     "0.53 M",  "yes"),
        ("Post-fusion TransformerEncoder ×2",  "1.58 M",  "yes"),
        ("MLPScoringHead  (4 × MLP)",          "0.09 M",  "yes"),
        ("─" * 40,                             "─" * 10,  "─" * 10),
        ("TOTAL",                              "~904 M",  "~303 M"),
    ]
    print("\nParameter estimates")
    print("=" * 65)
    for r in rows:
        print(f"  {r[0]:<40}  {r[1]:>10}  {r[2]:>10}")
    print("  * CNN + feature projection frozen by default")
    print("  † loaded from checkpoint but never called during scoring")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib flowchart
# ─────────────────────────────────────────────────────────────────────────────

def _plot(out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")
        return

    fig, ax = plt.subplots(figsize=(14, 22))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 22)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    C_AUDIO  = "#1f6feb"
    C_TEXT   = "#2ea043"
    C_FUSION = "#d29922"
    C_OUT    = "#bc4c00"
    C_TXT    = "#e6edf3"

    def box(x, y, w, h, label, sublabel="", color=C_AUDIO, fs=9):
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            linewidth=1.5, edgecolor=color,
            facecolor=color + "28",
        )
        ax.add_patch(rect)
        ax.text(x, y + (0.14 if sublabel else 0), label,
                ha="center", va="center", fontsize=fs,
                fontweight="bold", color=C_TXT)
        if sublabel:
            ax.text(x, y - 0.21, sublabel,
                    ha="center", va="center", fontsize=7.2,
                    color=color, style="italic")

    def arr(x1, y1, x2, y2, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.5, mutation_scale=12))

    def lbl(x, y, text, color="#888"):
        ax.text(x, y, text, ha="center", va="center",
                fontsize=7.2, color=color, style="italic",
                bbox=dict(boxstyle="round,pad=0.14", facecolor="#161b22",
                          edgecolor=color + "55", linewidth=0.8))

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(7, 21.5, "PronunciationScorer — Architecture",
            ha="center", va="center", fontsize=13,
            fontweight="bold", color=C_TXT)

    # ── AUDIO PATH  (left, x=3.5) ─────────────────────────────────────────
    AX = 3.5
    box(AX, 20.5, 5.5, 0.6, "Raw waveform", "[B, T_audio]", C_AUDIO)
    arr(AX, 20.2, AX, 19.75, C_AUDIO)

    # FlatHuBERT block — taller to show internal layers
    box(AX, 18.8, 5.8, 2.6,
        "FlatHuBERT  (pretrained)",
        "CNN f₀ → pos_conv+LN → TransformerEncoder ×24", C_AUDIO, 8.5)
    # Internal annotation
    ax.text(AX, 18.52, "flat — single resolution, no DOWN/UP",
            ha="center", va="center", fontsize=7, color=C_AUDIO + "cc",
            style="italic")

    lbl(AX, 17.7, "24 × [B, T', 1024]  hidden states", C_AUDIO)
    arr(AX, 17.95, AX, 17.4, C_AUDIO)

    box(AX, 17.1, 5.0, 0.55,
        "Learnable Weighted Sum",
        "softmax(wts) · Σ wᵢ·hᵢ → [B, T', 1024]", C_AUDIO, 8.5)
    arr(AX, 16.82, AX, 16.35, C_AUDIO)

    box(AX, 16.05, 4.0, 0.55, "Linear  1024 → 256", "", C_AUDIO)
    arr(AX, 15.82, AX, 15.35, C_AUDIO)

    box(AX, 15.05, 4.2, 0.55,
        "TransformerEncoder ×1",
        "pre-LN · 4 heads", C_AUDIO, 8.5)
    arr(AX, 14.8, AX, 14.25, C_AUDIO)
    lbl(AX, 14.0, "[B, T', 256]   Q", C_AUDIO)

    # ── TEXT PATH  (right, x=10.5) ────────────────────────────────────────
    TX = 10.5
    box(TX, 20.5, 5.5, 0.6, "Transcripts", "List[str]", C_TEXT)
    arr(TX, 20.2, TX, 19.75, C_TEXT)

    box(TX, 18.8, 5.5, 2.1,
        "Qwen3-Embedding-0.6B  (frozen)",
        "left-pad · instruct prefix · mean pool · L2", C_TEXT, 8.5)
    lbl(TX, 17.7, "[B, 1024]  float32", C_TEXT)
    arr(TX, 17.45, TX, 16.95, C_TEXT)

    box(TX, 16.65, 4.8, 0.55,
        "TextProjection",
        "Linear(1024→256) + LayerNorm + unsqueeze", C_TEXT, 8.5)
    arr(TX, 16.38, TX, 15.88, C_TEXT)
    lbl(TX, 15.65, "[B, 1, 256]   K / V", C_TEXT)

    # ── FUSION (center, x=7) ──────────────────────────────────────────────
    FX = 7.0

    arr(AX, 13.75, FX - 1.1, 12.85, C_FUSION)
    arr(TX, 15.45, FX + 1.1, 12.85, C_FUSION)

    box(FX, 12.55, 5.8, 0.75,
        "CrossAttentionFusion",
        "MHA(Q=audio, K=text, V=text) · 4h · residual · LN", C_FUSION, 8.5)
    lbl(FX, 11.9, "[B, T', 256]", C_FUSION)
    arr(FX, 12.17, FX, 11.6, C_FUSION)

    box(FX, 11.3, 5.2, 0.65,
        "TransformerEncoder ×2",
        "pre-LN · 4 heads", C_FUSION, 8.5)
    lbl(FX, 10.75, "[B, T', 256]", C_FUSION)
    arr(FX, 10.97, FX, 10.45, C_FUSION)

    box(FX, 10.15, 4.8, 0.6,
        "Masked Mean Pool  (over T')",
        "weighted by attention_mask", C_FUSION, 8.5)
    lbl(FX, 9.62, "[B, 256]", C_FUSION)
    arr(FX, 9.85, FX, 9.35, C_FUSION)

    box(FX, 9.0, 5.8, 1.1,
        "MLPScoringHead",
        "5 × MLP  —  Lin(256,128)→ReLU→Drop\n"
        "→Lin(128,64)→ReLU→Lin(64,1)→Sigmoid", C_OUT, 8.5)
    lbl(FX, 8.37, "[B, 5]  ∈ (0, 1)   × 10", C_OUT)
    arr(FX, 8.55, FX, 8.0, C_OUT)

    box(FX, 7.7, 5.8, 0.65,
        "Output  [B, 5]  ∈ (0, 10)  MOS",
        "total · accuracy · fluency · prosodic", C_OUT, 8.5)

    # ── Legend ─────────────────────────────────────────────────────────────
    for xi, (c, lbl_txt) in enumerate([(C_AUDIO, "Audio (FlatHuBERT)"),
                                        (C_TEXT,  "Text (Qwen3)"),
                                        (C_FUSION,"Fusion"),
                                        (C_OUT,   "Output")]):
        lx = 1.3 + xi * 2.85
        rect = FancyBboxPatch((lx - 0.8, 6.6), 1.6, 0.38,
                               boxstyle="round,pad=0.05",
                               linewidth=1.2, edgecolor=c,
                               facecolor=c + "35")
        ax.add_patch(rect)
        ax.text(lx, 6.79, lbl_txt, ha="center", va="center",
                fontsize=7.5, color=C_TXT)

    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Diagram saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# torchinfo summary
# ─────────────────────────────────────────────────────────────────────────────

def _torchinfo_summary(cfg: dict) -> None:
    try:
        import torchinfo
    except ImportError:
        print("torchinfo not installed — skipping (pip install torchinfo)")
        return

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from unittest.mock import patch
    from types import SimpleNamespace

    N_LAYERS = cfg["model"].get("num_hidden_layers", 24)
    H_DIM    = cfg["model"].get("speech_hidden_dim", 1024)

    # Stub FlatHuBERT — mirrors the real interface without loading weights
    class _FakeFlat(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self.feat_masking       = type("FM", (), {"mask_prob": 0.0})()
            self.feature_extractor  = nn.Conv1d(1, H_DIM, 10, 5)
            self.feature_projection = nn.Linear(H_DIM, H_DIM)
            self.layers             = nn.ModuleList(
                [nn.Linear(H_DIM, H_DIM) for _ in range(N_LAYERS)]
            )
            self._p = nn.Parameter(torch.zeros(1))

        def forward(self, wav, attn=None, apply_mask=False, output_hidden_states=False):
            B, T = wav.shape
            T2 = 50
            all_hs = [torch.zeros(B, T2, H_DIM) + self._p * 0
                      for _ in range(N_LAYERS)] if output_hidden_states else None
            return SimpleNamespace(all_hidden_states=all_hs,
                                   last_hidden=torch.zeros(B, T2, H_DIM))

        def _audio_mask_to_feat_mask(self, m, n):
            return torch.ones(m.shape[0], n, dtype=torch.bool)

        def load_state_dict(self, sd, strict=True): pass

    class _FakeQwen(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self._frozen = True
            self.linear  = nn.Linear(64, 1024)

        def forward(self, txts):
            return F.normalize(
                self.linear(torch.randn(len(txts), 64)).float(), p=2, dim=-1
            )

    from model.pronunciation_scorer import PronunciationScorer

    with patch("model.audio_encoder.FlatHuBERT",                _FakeFlat), \
         patch("model.pronunciation_scorer.Qwen3MeanPoolEncoder", _FakeQwen):
        model = PronunciationScorer(cfg)

    B, T = 2, 16000
    print(f"\ntorchinfo summary  (stub FlatHuBERT {N_LAYERS}L × {H_DIM}h):")
    torchinfo.summary(
        model,
        input_data=[
            torch.randn(B, T),
            torch.ones(B, T, dtype=torch.long),
            ["hello"] * B,
        ],
        depth=4,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PronunciationScorer")
    parser.add_argument("--no-plot",  action="store_true", help="Skip matplotlib diagram")
    parser.add_argument("--summary",  action="store_true", help="Show torchinfo summary")
    parser.add_argument("--out",      default="architecture.png", help="Output PNG path")
    parser.add_argument("--config",   default="configs/scoring_config.yaml")
    args = parser.parse_args()

    print(ASCII)
    _param_table()

    if args.summary:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        _torchinfo_summary(cfg)

    if not args.no_plot:
        _plot(args.out)


if __name__ == "__main__":
    main()
