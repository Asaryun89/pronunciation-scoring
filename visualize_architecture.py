"""
visualize_architecture.py

Print a detailed ASCII diagram and save a matplotlib flowchart of the
PronunciationScorer architecture (MultiResHuBERT-base + Qwen3 cross-attention)
after Upgrade 1: dimension-conditioned layer weights.

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
╔══════════════════════════════════════════════════════════════════════════════════╗
║   PronunciationScorer — Architecture (Upgrade 1: dim-conditioned layer weights) ║
║   Audio: MultiResHuBERT-base (4+4+4)     Text: Qwen3-Embedding-0.6B            ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  AUDIO PATH                               TEXT PATH                             ║
║  ─────────────────────────────────────    ─────────────────────────             ║
║  raw waveform  [B, T_audio]               transcripts  List[str]                ║
║         │                                       │                               ║
║  ┌──────▼────────────────────────────┐   ┌──────▼─────────────────────┐         ║
║  │  MultiResHuBERT  (pretrained)     │   │  Qwen3-Embedding-0.6B      │         ║
║  │  ─────────────────────────────    │   │  (frozen, 596 M params)    │         ║
║  │  f₀  CNN (frozen) [B, T', 768]    │   │  padding_side = "left"     │         ║
║  │       ConvFeatureMasking          │   │  instruct prefix            │         ║
║  │       pos_conv + LN + Dropout     │   │  mask-aware mean pool       │         ║
║  │  f₁  TransformerEncoder ×4 → H₁  │   │  L2 normalise               │         ║
║  │  ↓   DOWN  stride-2 DS-conv       │   └──────────────┬──────────────┘         ║
║  │  f₂  TransformerEncoder ×4 → H₂  │                  │  [B, 1024]             ║
║  │  ↑   UP   transposed + H₁ skip   │   ┌──────────────▼──────────────┐         ║
║  │  f₃  TransformerEncoder ×4 → H₃  │   │  TextProjection             │         ║
║  │  output_hidden_states = True      │   │  Linear(1024 → 256) + LN   │         ║
║  │  12 × [B, T', 768]               │   │  unsqueeze → [B, 1, 256]    │         ║
║  └──────────────┬────────────────────┘   └──────────────┬──────────────┘         ║
║                 │  12 × [B, T', 768]               [B, 1, 256]  (K / V)         ║
║                 │                                        │                       ║
║  ┌──────────────▼────────────────────────────────────────────────────────┐       ║
║  │   Dim-Conditioned Layer Weights   layer_weights  [4, 12]              │       ║
║  │   w = softmax(layer_weights, dim=1)                                   │       ║
║  │   dim_feats = einsum('dn,bntf→dbtf', w, stacked)  →  [4, B, T', 768] │       ║
║  │                                                                       │       ║
║  │   Row 0  accuracy : Gaussian peak ~ layer  3  (H1 high-res, early)   │       ║
║  │   Row 1  fluency  : Gaussian peak ~ layer  8  (H2/H3 boundary, mid)  │       ║
║  │   Row 2  prosodic : Gaussian peak ~ layer 10  (H3 reconstructed)     │       ║
║  │   Row 3  total    : uniform  (inherits pretrained 1-D weights)        │       ║
║  └──┬─────────────────┬─────────────────┬─────────────────┬─────────────┘       ║
║     │ dim 0           │ dim 1           │ dim 2           │ dim 3               ║
║     │ accuracy        │ fluency         │ prosodic        │ total               ║
║     │ [B, T', 768]    │ [B, T', 768]    │ [B, T', 768]    │ [B, T', 768]        ║
║     │                 │                 │                 │                     ║
║  ┌──▼─────────────────▼─────────────────▼─────────────────▼─────────────┐       ║
║  │   4 × Independent Linear(768 → 256)                                  │       ║
║  │   proj[0]         proj[1]          proj[2]         proj[3]           │       ║
║  └──┬─────────────────┬─────────────────┬─────────────────┬─────────────┘       ║
║     │                 │                 │                 │                     ║
║  ┌──▼─────────────────▼─────────────────▼─────────────────▼─────────────┐       ║
║  │   Shared Pre-Fusion TransformerEncoder ×1  (pre-LN · 4 heads)        │       ║
║  └──┬─────────────────┬─────────────────┬─────────────────┬─────────────┘       ║
║     │ pre_fused[0]    │ pre_fused[1]    │ pre_fused[2]    │ pre_fused[3]        ║
║     │ [B, T', 256]    │ [B, T', 256]    │ [B, T', 256]    │ [B, T', 256]        ║
║     │                 │                 │                 │                     ║
║  Masked             Masked            Masked             └──────────────────────┐║
║  Mean Pool          Mean Pool         Mean Pool                                 │║
║     │ [B, 256]        │ [B, 256]        │ [B, 256]                    (K/V from)│║
║     │                 │                 │              ┌──────────────▼──────────┘║
║     │                 │                 │              │  CrossAttentionFusion   ║
║     │                 │                 │              │  MHA(Q=tot, K=kv, V=kv) ║
║     │                 │                 │              │  4 heads · res · LN     ║
║     │                 │                 │              └──────────────┬──────────║
║     │                 │                 │                             │           ║
║     │                 │                 │              ┌──────────────▼──────────┐║
║     │                 │                 │              │  TransformerEncoder ×2  │║
║     │                 │                 │              │  pre-LN · 4 heads       │║
║     │                 │                 │              └──────────────┬──────────┘║
║     │                 │                 │                             │           ║
║     │                 │                 │              ┌──────────────▼──────────┐║
║     │                 │                 │              │  Masked Mean Pool (tot) │║
║     │                 │                 │              └──────────────┬──────────┘║
║     │                 │                 │                    [B, 256] │           ║
║     ▼                 ▼                 ▼                             ▼           ║
║  accuracy_mlp     fluency_mlp      prosodic_mlp               total_mlp         ║
║  Lin(256,128)     Lin(256,128)     Lin(256,128)               Lin(256,128)      ║
║  →ReLU→Drop       →ReLU→Drop       →ReLU→Drop                 →ReLU→Drop       ║
║  →Lin(128,64)     →Lin(128,64)     →Lin(128,64)               →Lin(128,64)     ║
║  →ReLU→Sigmoid    →ReLU→Sigmoid    →ReLU→Sigmoid              →ReLU→Sigmoid    ║
║     │ [B,1]          │ [B,1]           │ [B,1]                      │ [B,1]     ║
║     └────────────────┴─────────────────┴────────────────────────────┘           ║
║                                        │                                         ║
║                                 [B, 4] ∈ (0, 1)                                 ║
║                         [total, accuracy, fluency, prosodic]                    ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Optimizer param groups:                                                         ║
║    A  MultiResHuBERT backbone (trainable layers)              lr = 5e-5         ║
║    B  layer_weights [4,12] + proj[0-3] + pre-transformer      lr = 1e-4         ║
║    C  text encoder (Qwen3, frozen)                            lr = 0.0          ║
║    D  fusion + post-transformer + scoring head                lr = 1e-4         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# Parameter count table
# ─────────────────────────────────────────────────────────────────────────────

def _param_table() -> None:
    rows = [
        ("Component",                               "Params",    "Trainable"),
        ("─" * 45,                                  "─" * 10,   "─" * 10),
        ("MultiResHuBERT-base  (hubert-base-ls960)", "~94 M",   "~88 M*"),
        ("  f₀  CNN feature extractor",              "~4.7 M",  "frozen"),
        ("  f₀  feature projection",                 "~0.6 M",  "frozen"),
        ("  pos_conv + LayerNorm",                   "~0.3 M",  "yes"),
        ("  f₁  TransformerEncoder ×4  (high-res)",  "~28 M",   "yes"),
        ("  DOWN  depthwise-sep conv",               "~0.5 M",  "yes"),
        ("  f₂  TransformerEncoder ×4  (low-res)",   "~28 M",   "yes"),
        ("  UP   transposed conv + gated skip",      "~0.5 M",  "yes"),
        ("  f₃  TransformerEncoder ×4  (high-res)",  "~28 M",   "yes"),
        ("  UnitPredictionHeads  (unused at score)",  "~0.2 M",  "frozen†"),
        ("─" * 45,                                  "─" * 10,   "─" * 10),
        ("Dim-cond. layer weights  [4 × 12]",        "< 0.1 K", "yes"),
        ("4 × Linear proj  768 → 256",              "0.79 M",  "yes"),
        ("Shared Pre-fusion TransformerEncoder ×1",  "1.31 M",  "yes"),
        ("─" * 45,                                  "─" * 10,   "─" * 10),
        ("Qwen3-Embedding-0.6B",                    "596 M",   "frozen"),
        ("TextProjection  1024 → 256",              "0.26 M",  "yes"),
        ("CrossAttentionFusion  (MHA 4h)",           "0.26 M",  "yes"),
        ("Post-fusion TransformerEncoder ×2",        "2.62 M",  "yes"),
        ("MLPScoringHead  (4 × MLP, named sub-heads)", "0.16 M", "yes"),
        ("─" * 45,                                  "─" * 10,   "─" * 10),
        ("TOTAL",                                   "~695 M",  "~94 M"),
    ]
    print("\nParameter estimates")
    print("=" * 72)
    for r in rows:
        print(f"  {r[0]:<45}  {r[1]:>10}  {r[2]:>10}")
    print("  * CNN + feature projection frozen by default")
    print("  † loaded from pretrain checkpoint; not called during scoring")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib flowchart
# ─────────────────────────────────────────────────────────────────────────────

def _plot(out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")
        return

    FW, FH = 18, 30
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    C_AUDIO  = "#1f6feb"   # blue
    C_TEXT   = "#2ea043"   # green
    C_SPLIT  = "#8957e5"   # purple — new dim-conditioned layer weights
    C_FUSION = "#d29922"   # amber
    C_OUT    = "#bc4c00"   # orange
    C_TXT    = "#e6edf3"   # near-white text
    C_DIM    = ["#58a6ff", "#3fb950", "#d2a8ff", "#ffa657"]  # per-dimension lane colors

    def box(x, y, w, h, label, sublabel="", color=C_AUDIO, fs=9):
        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.12",
            linewidth=1.5, edgecolor=color,
            facecolor=color + "22",
        )
        ax.add_patch(rect)
        ax.text(x, y + (0.16 if sublabel else 0), label,
                ha="center", va="center", fontsize=fs,
                fontweight="bold", color=C_TXT)
        if sublabel:
            ax.text(x, y - 0.22, sublabel,
                    ha="center", va="center", fontsize=7,
                    color=color, style="italic")

    def arr(x1, y1, x2, y2, color="#555", lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=12))

    def lbl(x, y, text, color="#888", fs=7):
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fs, color=color, style="italic",
                bbox=dict(boxstyle="round,pad=0.14", facecolor="#161b22",
                          edgecolor=color + "55", linewidth=0.8))

    def hline(x1, x2, y, color="#555", lw=1.2, dashed=False):
        ls = "--" if dashed else "-"
        ax.plot([x1, x2], [y, y], color=color, lw=lw, ls=ls, zorder=2)

    def vline(x, y1, y2, color="#555", lw=1.2, dashed=False):
        ls = "--" if dashed else "-"
        ax.plot([x, x], [y1, y2], color=color, lw=lw, ls=ls, zorder=2)

    # ── Title ──────────────────────────────────────────────────────────────────
    ax.text(FW / 2, 29.5,
            "PronunciationScorer — Architecture (Upgrade 1)",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=C_TXT)
    ax.text(FW / 2, 29.0,
            "dim-conditioned layer weights  ·  4 × parallel audio paths  ·  per-dimension MLP heads",
            ha="center", va="center", fontsize=8.5, color="#8b949e")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # AUDIO PATH  (left, centred x=4.5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    AX = 4.5

    box(AX, 28.1, 6.0, 0.65, "Raw waveform", "[B, T_audio]", C_AUDIO)
    arr(AX, 27.77, AX, 27.22, C_AUDIO)

    # MultiResHuBERT — large block showing internal MR structure
    box(AX, 24.9, 6.2, 4.4,
        "MultiResHuBERT  (pretrained backbone)",
        "facebook/hubert-base-ls960", C_AUDIO, 8.5)
    for dy, txt in [
        ( 1.5, "f₀  CNN (frozen)  → [B, T', 768]"),
        ( 1.0, "    ConvFeatureMasking + pos_conv + LN + Dropout"),
        ( 0.5, "f₁  TransformerEncoder ×4  (high-res)  →  H₁"),
        ( 0.0, "↓   DOWN  stride-2 depthwise-sep convolution"),
        (-0.5, "f₂  TransformerEncoder ×4  (low-res)   →  H₂"),
        (-1.0, "↑   UP    transposed conv + gated H₁ skip"),
        (-1.5, "f₃  TransformerEncoder ×4  (high-res)  →  H₃"),
    ]:
        ax.text(AX, 24.9 + dy, txt,
                ha="center", va="center", fontsize=6.8,
                color=C_AUDIO + "cc", style="italic")

    lbl(AX, 22.35, "output_hidden_states=True  →  12 × [B, T', 768]", C_AUDIO, 7)
    arr(AX, 22.68, AX, 22.05, C_AUDIO)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TEXT PATH  (right, centred x=13.5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TX = 13.5

    box(TX, 28.1, 5.5, 0.65, "Transcripts", "List[str]", C_TEXT)
    arr(TX, 27.77, TX, 27.22, C_TEXT)

    box(TX, 25.3, 5.5, 3.3,
        "Qwen3-Embedding-0.6B  (frozen)",
        "left-pad · instruct prefix · mean pool · L2", C_TEXT, 8.5)
    for dy, txt in [
        ( 0.9, "padding_side = 'left'  (decoder-only last-token)"),
        ( 0.4, "Think() suppressed, instruct prefix prepended"),
        (-0.1, "mask-aware mean pooling across token dim"),
        (-0.6, "L2 normalise  →  [B, 1024]"),
    ]:
        ax.text(TX, 25.3 + dy, txt,
                ha="center", va="center", fontsize=6.6,
                color=C_TEXT + "cc", style="italic")

    lbl(TX, 23.45, "[B, 1024]  float32", C_TEXT, 7)
    arr(TX, 23.65, TX, 23.12, C_TEXT)

    box(TX, 22.7, 5.5, 0.7,
        "TextProjection",
        "Linear(1024 → 256) + LayerNorm + unsqueeze", C_TEXT, 8.5)
    lbl(TX, 22.12, "[B, 1, 256]   K / V", C_TEXT, 7)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DIM-CONDITIONED LAYER WEIGHTS  (full width, centred x=9)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    LWY = 20.4
    box(9, LWY, 17.0, 2.1,
        "Dim-Conditioned Layer Weights   layer_weights  [4, 12]",
        "", C_SPLIT, 9.5)

    row_info = [
        (0, "accuracy",  "peak ~ layer  3", "H1 high-res, early"),
        (1, "fluency",   "peak ~ layer  8", "H2/H3 boundary, mid"),
        (2, "prosodic",  "peak ~ layer 10", "H3 reconstructed, late"),
        (3, "total",     "uniform init",    "inherits pretrained 1-D weights"),
    ]
    for i, (ridx, rname, peak, desc) in enumerate(row_info):
        rx = 1.5 + i * 4.0
        c  = C_DIM[i]
        ax.text(rx, LWY + 0.55, f"Row {ridx}  ({rname})",
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=c)
        ax.text(rx, LWY + 0.18, peak,
                ha="center", va="center", fontsize=6.8,
                color=c, style="italic")
        ax.text(rx, LWY - 0.22, desc,
                ha="center", va="center", fontsize=6.5,
                color=c + "bb", style="italic")

    ax.text(9, LWY - 0.72,
            "w = softmax(layer_weights, dim=1)   dim_feats = einsum('dn,bntf→dbtf', w, stacked)",
            ha="center", va="center", fontsize=7.5,
            color=C_SPLIT + "cc", style="italic")

    # Arrow from backbone to layer weights
    arr(AX, 22.05, 3.5, LWY + 1.05, C_AUDIO)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4 lane x-positions
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    LX = [1.8, 5.6, 9.4, 13.2]   # accuracy, fluency, prosodic, total

    # Vertical arrows from layer_weights to proj block
    LWB = LWY - 1.05
    for i, lx in enumerate(LX):
        arr(lx, LWB, lx, LWB - 0.45, C_DIM[i])
        lbl(lx, LWB - 0.55, "[B, T', 768]", C_DIM[i], 6.5)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4 × Linear(768 → 256) — wide block
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PRJY = 18.5
    box(9, PRJY, 17.0, 0.8,
        "4 × Independent Linear(768 → 256)",
        "proj[0]  ·  proj[1]  ·  proj[2]  ·  proj[3]", C_SPLIT, 9)

    for i, lx in enumerate(LX):
        arr(lx, PRJY + 0.4, lx, PRJY - 0.4, C_DIM[i])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Shared pre-fusion TransformerEncoder — wide block
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PTY = 17.2
    box(9, PTY, 17.0, 0.75,
        "Shared Pre-Fusion TransformerEncoder ×1  (pre-LN · 4 heads)",
        "same module applied per-dimension; weights not replicated", C_SPLIT, 8.5)

    for i, lx in enumerate(LX):
        arr(lx, PRJY - 0.4, lx, PTY + 0.375, C_DIM[i])
        arr(lx, PTY - 0.375, lx, PTY - 0.75, C_DIM[i])
        lbl(lx, PTY - 0.9, "pre_fused[%d]\n[B, T', 256]" % i, C_DIM[i], 6.2)

    # Labels under pre-transformer
    dim_names = ["accuracy", "fluency", "prosodic", "total"]
    for i, (lx, name) in enumerate(zip(LX, dim_names)):
        ax.text(lx, PTY - 1.25, name,
                ha="center", va="center", fontsize=7,
                fontweight="bold", color=C_DIM[i])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Dims 0-2: Masked mean pool → MLP sub-heads
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    POOLY = 14.8
    for i in range(3):
        lx = LX[i]
        c  = C_DIM[i]
        arr(lx, PTY - 1.45, lx, POOLY + 0.35, c)
        box(lx, POOLY, 3.0, 0.6, "Masked Mean Pool", "feat_mask · [B, 256]", c, 7.5)
        arr(lx, POOLY - 0.3, lx, POOLY - 0.7, c)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Dim 3 (total): CrossAttentionFusion path
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CX = LX[3]   # x=13.2

    # Arrow: TextProjection K/V → CrossAttentionFusion
    arr(TX, 22.12 - 0.2, CX + 0.5, 15.85, C_TEXT, lw=1.2)
    lbl(TX - 0.3, 19.0, "K / V →", C_TEXT, 6.5)

    arr(CX, PTY - 1.45, CX, 16.15, C_DIM[3])
    box(CX, 15.7, 4.5, 0.85,
        "CrossAttentionFusion",
        "MHA(Q=total, K=text, V=text) · 4h · res · LN", C_FUSION, 8)
    arr(CX, 15.27, CX, 14.77, C_FUSION)

    box(CX, 14.45, 4.0, 0.6,
        "TransformerEncoder ×2",
        "pre-LN · 4 heads", C_FUSION, 8)
    arr(CX, 14.15, CX, 13.65, C_FUSION)

    box(CX, 13.35, 3.8, 0.6,
        "Masked Mean Pool  (total)",
        "feat_mask · [B, 256]", C_FUSION, 8)
    arr(CX, 13.05, CX, 12.5, C_FUSION)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MLP sub-heads
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    HEAD_LABEL = ["accuracy_mlp", "fluency_mlp", "prosodic_mlp", "total_mlp"]
    HEAD_Y     = [13.35, 13.35, 13.35, 12.5]   # total starts lower (after pool)
    MLP_Y      = [11.7,  11.7,  11.7,  11.0]

    for i in range(3):
        lx = LX[i]
        c  = C_DIM[i]
        hy = HEAD_Y[i]
        my = MLP_Y[i]
        # pool already drawn; arrow from pool bottom to MLP head
        box(lx, my, 3.0, 1.2,
            HEAD_LABEL[i],
            "Lin(256,128)→ReLU→Drop\n→Lin(128,64)→ReLU→Lin(64,1)→Sigmoid",
            c, 7.5)
        arr(lx, my + 0.6, lx, my - 0.6, c)
        lbl(lx, my - 0.85, "[B, 1]", c, 6.5)

    # total MLP
    box(CX, MLP_Y[3], 3.6, 1.2,
        "total_mlp",
        "Lin(256,128)→ReLU→Drop\n→Lin(128,64)→ReLU→Lin(64,1)→Sigmoid",
        C_DIM[3], 7.5)
    arr(CX, MLP_Y[3] + 0.6, CX, MLP_Y[3] - 0.6, C_DIM[3])
    lbl(CX, MLP_Y[3] - 0.85, "[B, 1]", C_DIM[3], 6.5)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Merge → output
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    MERGE_Y = 9.3
    MID_X   = (LX[0] + LX[3]) / 2   # ≈ 7.5

    for i, (lx, my) in enumerate(zip(LX, MLP_Y)):
        # vertical down from MLP bottom
        bot = my - 1.2
        vline(lx, bot, MERGE_Y + 0.1, C_DIM[i])
        ax.annotate("", xy=(lx, MERGE_Y + 0.1), xytext=(lx, bot),
                    arrowprops=dict(arrowstyle="->", color=C_DIM[i], lw=1.2))

    # horizontal merge bar
    hline(LX[0], LX[3], MERGE_Y, C_OUT, lw=1.8)
    arr(MID_X, MERGE_Y, MID_X, MERGE_Y - 0.5, C_OUT, lw=1.8)

    box(MID_X, MERGE_Y - 0.9, 6.5, 0.65,
        "Output  [B, 4]  ∈ (0, 1)",
        "torch.cat([score_tot, score_acc, score_flu, score_pro], dim=1)", C_OUT, 8.5)

    lbl(MID_X, MERGE_Y - 1.65,
        "order: [total, accuracy, fluency, prosodic]  —  matches SCORE_DIMS", C_OUT, 7)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Legend + optimizer groups
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    legend_items = [
        (C_AUDIO,  "Audio (MultiResHuBERT)"),
        (C_TEXT,   "Text (Qwen3)"),
        (C_SPLIT,  "Dim-cond. layer weights"),
        (C_FUSION, "CrossAttn / Fusion"),
        (C_OUT,    "Output"),
    ]
    for xi, (c, ltxt) in enumerate(legend_items):
        lx2 = 1.0 + xi * 3.4
        rect = FancyBboxPatch((lx2 - 1.5, 7.15), 3.0, 0.42,
                               boxstyle="round,pad=0.06",
                               linewidth=1.2, edgecolor=c,
                               facecolor=c + "35")
        ax.add_patch(rect)
        ax.text(lx2, 7.36, ltxt, ha="center", va="center",
                fontsize=7.5, color=C_TXT)

    # Optimizer groups table
    opt_rows = [
        ("A", "MultiResHuBERT backbone (trainable layers)", "lr = 5e-5",  C_AUDIO),
        ("B", "layer_weights [4,12] + proj[0-3] + pre-transformer", "lr = 1e-4", C_SPLIT),
        ("C", "text encoder (Qwen3, frozen)", "lr = 0.0",  C_TEXT),
        ("D", "fusion + post-transformer + scoring head", "lr = 1e-4", C_FUSION),
    ]
    ax.text(FW / 2, 6.5, "Optimizer parameter groups", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=C_TXT)
    for ri, (grp, desc, lr_txt, gc) in enumerate(opt_rows):
        ry = 6.0 - ri * 0.45
        ax.text(1.0, ry, f"Group {grp}", ha="left", va="center",
                fontsize=7.5, fontweight="bold", color=gc)
        ax.text(3.5, ry, desc, ha="left", va="center", fontsize=7.2, color=C_TXT)
        ax.text(FW - 0.5, ry, lr_txt, ha="right", va="center",
                fontsize=7.2, color=gc)

    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Diagram saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# torchinfo summary
# ─────────────────────────────────────────────────────────────────────────────

def _torchinfo_summary(cfg: dict) -> None:
    try:
        import torchinfo
    except ImportError:
        print("torchinfo not installed — skipping (pip install torchinfo)")
        return

    import copy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from types import SimpleNamespace
    from unittest.mock import patch

    mcfg     = cfg["model"]
    N_LAYERS = mcfg.get("num_hidden_layers", 12)
    H_DIM    = mcfg.get("speech_hidden_dim", 768)
    N_F1     = mcfg.get("f1_layer_count", 4)
    N_F2     = mcfg.get("f2_layer_count", 4)
    N_F3     = mcfg.get("f3_layer_count", 4)

    # Stub MultiResHuBERT — mirrors the interface AudioEncoder needs without
    # loading actual HuBERT weights (fast startup for shape inspection).
    class _FakeMultiRes(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self.feat_masking       = type("FM", (), {"mask_prob": 0.0})()
            self.feature_extractor  = nn.Conv1d(1, H_DIM, 10, 5)
            self.feature_projection = nn.Linear(H_DIM, H_DIM)
            self.f1_layers = nn.ModuleList(
                [nn.Linear(H_DIM, H_DIM) for _ in range(N_F1)]
            )
            self.f2_layers = nn.ModuleList(
                [nn.Linear(H_DIM, H_DIM) for _ in range(N_F2)]
            )
            self.f3_layers = nn.ModuleList(
                [nn.Linear(H_DIM, H_DIM) for _ in range(N_F3)]
            )
            self._p = nn.Parameter(torch.zeros(1))

        def forward(self, wav, attn=None, apply_mask=False,
                    output_hidden_states=False):
            B = wav.shape[0]
            T2 = 50
            zero = self._p * 0
            all_hs = (
                [torch.zeros(B, T2, H_DIM) + zero for _ in range(N_LAYERS)]
                if output_hidden_states else None
            )
            return SimpleNamespace(all_hidden_states=all_hs)

        def _audio_mask_to_feat_mask(self, mask, n):
            return torch.ones(mask.shape[0], n, dtype=torch.bool)

        def load_state_dict(self, sd, strict=True):
            return [], []

    class _FakeQwen(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self.linear = nn.Linear(64, 1024)

        def forward(self, txts):
            return F.normalize(
                self.linear(torch.randn(len(txts), 64)).float(), p=2, dim=-1
            )

        def parameters(self, recurse=True):
            return iter([])

    cfg_stub = copy.deepcopy(cfg)
    cfg_stub["model"]["pretrain_checkpoint"] = None

    from model.pronunciation_scorer import PronunciationScorer

    with patch("model.audio_encoder.MultiResHuBERT",             _FakeMultiRes), \
         patch("model.pronunciation_scorer.Qwen3MeanPoolEncoder", _FakeQwen):
        model = PronunciationScorer(cfg_stub)

    B, T = 2, 16000
    print(f"\ntorchinfo summary  (stub MultiResHuBERT {N_LAYERS}L × {H_DIM}h):")
    torchinfo.summary(
        model,
        input_data=[
            torch.randn(B, T),
            torch.ones(B, T, dtype=torch.long),
            ["hello world"] * B,
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
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib diagram")
    parser.add_argument("--summary", action="store_true", help="Show torchinfo summary")
    parser.add_argument("--out",     default="architecture.png", help="Output PNG path")
    parser.add_argument("--config",  default="configs/scoring_config.yaml")
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
