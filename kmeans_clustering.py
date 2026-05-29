#!/usr/bin/env python3
"""
Standalone k-means clustering on pre-extracted feature arrays.

This script operates on .npy files produced by ``extract_features.py`` and
is distinct from ``training/kmeans.py`` which extracts features on-the-fly.
Use this for:

  • Generating HuBERT pre-training targets from a specific encoder layer
  • Probing experiments (cluster structure of H1 / H2 / H3)
  • Selecting the best K (comparing inertia / silhouette across K values)

Workflow
────────
    # 1. Extract features from the training corpus
    python extract_features.py \\
        --config checkpoints/pretrain/config.yaml \\
        --checkpoint checkpoints/pretrain/final_pretrain.pt \\
        --wav-scp data/train/wav.scp \\
        --output-dir features/train/ \\
        --layers h1,h2,h3

    # 2. Fit k-means for several K values on the h1 layer
    python kmeans_clustering.py \\
        --features-dir features/train/ \\
        --layer h1 \\
        --k 100,200,500 \\
        --output-dir data/kmeans/

    # 3. Fit only K=100 and compute per-utterance labels (pre-training targets)
    python kmeans_clustering.py \\
        --features-dir features/train/ \\
        --layer h1 \\
        --k 100 \\
        --output-dir data/kmeans/ \\
        --save-labels \\
        --plot
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_features(
    features_dir: Path,
    layer:        str,
    max_utts:     Optional[int] = None,
    l2_norm:      bool          = True,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Load all ``*.{layer}.npy`` files and concatenate into a single array.

    Args:
        features_dir: Directory with .npy files from extract_features.py.
        layer:        Layer name: "h0", "h1", "h2", or "h3".
        max_utts:     Limit number of utterances (useful for quick testing).
        l2_norm:      L2-normalise each frame (recommended for k-means).

    Returns:
        frames:    (N_frames, H)         — concatenated feature vectors
        utt_ids:   List of utterance IDs — one per file
        lengths:   List of frame counts  — one per utterance
    """
    files = sorted(features_dir.glob(f"*.{layer}.npy"))
    if not files:
        raise FileNotFoundError(
            f"No *.{layer}.npy files found in {features_dir}\n"
            "Run extract_features.py first."
        )
    if max_utts is not None:
        files = files[:max_utts]

    all_frames: List[np.ndarray] = []
    utt_ids:    List[str]        = []
    lengths:    List[int]        = []

    for p in files:
        arr = np.load(p).astype(np.float32)   # (T, H)
        if l2_norm:
            arr = normalize(arr, norm="l2")
        all_frames.append(arr)
        utt_ids.append(p.name.replace(f".{layer}.npy", ""))
        lengths.append(arr.shape[0])

    frames = np.concatenate(all_frames, axis=0)   # (N_total, H)
    log.info(
        "Loaded %d utterances  (%d frames, dim=%d)  layer=%s",
        len(utt_ids), frames.shape[0], frames.shape[1], layer,
    )
    return frames, utt_ids, lengths


# ─────────────────────────────────────────────────────────────────────────────
# K-means fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_kmeans(
    features:    np.ndarray,   # (N, H)
    k:           int,
    n_init:      int   = 5,
    batch_size:  int   = 10_000,
    max_iter:    int   = 150,
    seed:        int   = 42,
    verbose:     bool  = True,
) -> MiniBatchKMeans:
    """
    Fit MiniBatchKMeans with ``k`` clusters.

    Returns a fitted sklearn MiniBatchKMeans object.
    """
    log.info("Fitting MiniBatchKMeans  K=%d  frames=%d …", k, features.shape[0])
    t0 = time.perf_counter()

    km = MiniBatchKMeans(
        n_clusters  = k,
        n_init      = n_init,
        batch_size  = min(batch_size, features.shape[0]),
        max_iter    = max_iter,
        random_state= seed,
        verbose     = int(verbose),
    )
    km.fit(features)

    elapsed = time.perf_counter() - t0
    log.info(
        "  K=%d  inertia=%.2f  time=%.1fs", k, km.inertia_, elapsed
    )
    return km


# ─────────────────────────────────────────────────────────────────────────────
# Statistics and diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def cluster_stats(
    labels:  np.ndarray,   # (N_frames,)
    k:       int,
    top_n:   int = 10,
) -> Dict[str, object]:
    """
    Compute cluster-size distribution statistics.

    Returns a dict with:
      counts:       array of cluster sizes (k,)
      min_size:     smallest cluster
      max_size:     largest cluster
      mean_size:    average cluster size
      std_size:     std of cluster sizes
      gini:         Gini coefficient (0=uniform, 1=concentrated)
      top_clusters: list of (cluster_id, count) for the top_n biggest clusters
    """
    counts = np.bincount(labels, minlength=k).astype(float)
    sorted_counts = np.sort(counts)

    # Gini coefficient
    n = len(counts)
    gini = (2 * np.sum(sorted_counts * np.arange(1, n + 1)) - (n + 1) * np.sum(sorted_counts)) / (
        n * np.sum(sorted_counts) + 1e-12
    )

    top_idx = np.argsort(counts)[::-1][:top_n]
    return {
        "counts":       counts,
        "min_size":     int(counts.min()),
        "max_size":     int(counts.max()),
        "mean_size":    float(counts.mean()),
        "std_size":     float(counts.std()),
        "gini":         float(gini),
        "top_clusters": [(int(i), int(counts[i])) for i in top_idx],
    }


def print_stats_table(k: int, stats: Dict[str, object], inertia: float) -> None:
    print(f"\n── K={k} ─────────────────────────────────────────────────")
    print(f"  Inertia       : {inertia:,.2f}")
    print(f"  Cluster sizes : min={stats['min_size']}  max={stats['max_size']}"
          f"  mean={stats['mean_size']:.1f}  std={stats['std_size']:.1f}")
    print(f"  Gini coeff    : {stats['gini']:.4f}  (0=uniform  1=concentrated)")
    top = "  ".join(f"#{i}:{c}" for i, c in stats["top_clusters"][:5])
    print(f"  Top-5 clusters: {top}")


# ─────────────────────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────────────────────

def save_kmeans_artifacts(
    km:         MiniBatchKMeans,
    labels:     np.ndarray,
    utt_ids:    List[str],
    lengths:    List[int],
    k:          int,
    layer:      str,
    output_dir: Path,
    save_labels: bool = False,
) -> None:
    """
    Save:
      • ``centroids_k{k}_{layer}.npy``   — (k, H) centroid matrix
      • ``kmeans_k{k}_{layer}.pkl``      — full fitted MiniBatchKMeans object
      • ``labels_k{k}_{layer}.pkl``      — {utt_id: np.int16 label array}
                                            (only when save_labels=True)
    """
    prefix = f"k{k}_{layer}"

    # Centroids
    centroid_path = output_dir / f"centroids_{prefix}.npy"
    np.save(centroid_path, km.cluster_centers_.astype(np.float32))
    log.info("Centroids saved → %s  shape=%s", centroid_path.name, km.cluster_centers_.shape)

    # Full k-means object
    km_path = output_dir / f"kmeans_{prefix}.pkl"
    with open(km_path, "wb") as f:
        pickle.dump(km, f, protocol=4)
    log.info("K-means model saved → %s", km_path.name)

    if not save_labels:
        return

    # Per-utterance label arrays (compatible with PretrainDataset format)
    offset = 0
    utt_labels: Dict[str, np.ndarray] = {}
    for uid, length in zip(utt_ids, lengths):
        utt_labels[uid] = labels[offset: offset + length].astype(np.int16)
        offset += length

    label_path = output_dir / f"labels_{prefix}.pkl"
    with open(label_path, "wb") as f:
        pickle.dump(utt_labels, f, protocol=4)
    log.info("Labels saved → %s  (%d utterances)", label_path.name, len(utt_labels))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    all_k:       List[int],
    all_inertia: List[float],
    all_stats:   List[Dict],
    output_dir:  Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plots.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Elbow curve
    axes[0].plot(all_k, all_inertia, "o-", color="steelblue", linewidth=2)
    axes[0].set_xlabel("K (number of clusters)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Curve")
    axes[0].grid(True, alpha=0.3)

    # Gini coefficients
    ginis = [s["gini"] for s in all_stats]
    axes[1].bar(range(len(all_k)), ginis, color="salmon")
    axes[1].set_xticks(range(len(all_k)))
    axes[1].set_xticklabels([f"K={k}" for k in all_k])
    axes[1].set_ylabel("Gini coefficient")
    axes[1].set_title("Cluster Balance (lower = more uniform)")
    axes[1].set_ylim(0, 1)

    # Cluster size distribution for the last (largest) K
    last_stats = all_stats[-1]
    counts_sorted = np.sort(last_stats["counts"])[::-1]
    axes[2].bar(range(len(counts_sorted)), counts_sorted, color="mediumpurple", width=1.0)
    axes[2].set_xlabel(f"Cluster index (sorted, K={all_k[-1]})")
    axes[2].set_ylabel("Number of frames")
    axes[2].set_title(f"Cluster Size Distribution  K={all_k[-1]}")

    plt.suptitle("K-means Clustering Analysis", fontsize=13)
    plt.tight_layout()

    out_path = output_dir / "kmeans_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Analysis plot saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-means clustering on pre-extracted features"
    )
    parser.add_argument("--features-dir", required=True,
                        help="Directory with .npy files from extract_features.py")
    parser.add_argument("--layer",        default="h1",
                        choices=["h0", "h1", "h2", "h3"],
                        help="Which encoder layer to cluster (default: h1)")
    parser.add_argument("--k",            default="100,200,500",
                        help="Comma-separated K values (default: 100,200,500)")
    parser.add_argument("--output-dir",   default="data/kmeans/")
    parser.add_argument("--n-init",       type=int, default=5,
                        help="Number of k-means initialisations")
    parser.add_argument("--batch-size",   type=int, default=10_000,
                        help="MiniBatchKMeans batch size")
    parser.add_argument("--max-utts",     type=int, default=None,
                        help="Limit utterances loaded (for quick tests)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--no-l2-norm",   action="store_true",
                        help="Skip L2 normalisation of features before clustering")
    parser.add_argument("--save-labels",  action="store_true",
                        help="Save per-utterance cluster label files (needed for pre-training)")
    parser.add_argument("--plot",         action="store_true",
                        help="Save elbow and distribution plots (requires matplotlib)")
    args = parser.parse_args()

    # Parse K values
    k_values = [int(x.strip()) for x in args.k.split(",")]
    if not k_values:
        parser.error("--k must be a comma-separated list of integers")

    features_dir = Path(args.features_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load features (once for all K values) ─────────────────────────────
    features, utt_ids, lengths = load_features(
        features_dir,
        layer    = args.layer,
        max_utts = args.max_utts,
        l2_norm  = not args.no_l2_norm,
    )

    all_inertia: List[float]      = []
    all_stats:   List[Dict]       = []

    # ── fit k-means for each K ─────────────────────────────────────────────
    for k in k_values:
        km     = fit_kmeans(
            features,
            k          = k,
            n_init     = args.n_init,
            batch_size = args.batch_size,
            seed       = args.seed,
        )
        labels = km.labels_.astype(np.int32)
        stats  = cluster_stats(labels, k)

        print_stats_table(k, stats, km.inertia_)
        all_inertia.append(km.inertia_)
        all_stats.append(stats)

        save_kmeans_artifacts(
            km, labels, utt_ids, lengths,
            k           = k,
            layer       = args.layer,
            output_dir  = output_dir,
            save_labels = args.save_labels,
        )

    # ── summary across K values ────────────────────────────────────────────
    print("\n── Summary ────────────────────────────────────────────────")
    print(f"  {'K':>6}  {'Inertia':>14}  {'Gini':>8}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*8}")
    for k, inertia, stats in zip(k_values, all_inertia, all_stats):
        print(f"  {k:>6}  {inertia:>14,.2f}  {stats['gini']:>8.4f}")

    if args.plot:
        plot_results(k_values, all_inertia, all_stats, output_dir)

    log.info(
        "Done.  Artifacts saved to %s\n"
        "Next step: use --save-labels output as labels_path in configs/pretrain.yaml",
        output_dir,
    )


if __name__ == "__main__":
    main()
