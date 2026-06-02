"""
K-means quantiser for HuBERT-style self-supervised pre-training.

Features are sourced from the HuggingFace dataset (mispeech/speechocean762).
Two feature sources are supported:

  Iteration 1 — MFCC features
      No pretrained model required.  MFCCs are extracted at HuBERT's native
      20 ms frame rate (hop_length = 320 samples at 16 kHz) so the cluster
      labels align one-to-one with CNN feature frames.

  Iteration 2 — model internal features
      Extract the output of f₁ (high-resolution encoder) from a checkpoint
      trained in iteration 1.  This gives semantically richer units.

Typical workflow
────────────────
    from datasets import load_dataset, Audio
    hf_ds = load_dataset("mispeech/speechocean762", split="train")
    hf_ds = hf_ds.cast_column("audio", Audio(sampling_rate=16_000))

    q = KMeansQuantizer(n_clusters_hi=100)
    q.fit_from_mfcc(hf_ds)
    q.assign_and_save("data/kmeans/labels_iter1.pkl", downsample_stride=2)
    q.save("data/kmeans/kmeans_iter1.pkl")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torchaudio.transforms as T
from torch import Tensor

log = logging.getLogger(__name__)

_HUBERT_CNN_STRIDE = 320
_TARGET_SR         = 16_000


# ─────────────────────────────────────────────────────────────────────────────
# Feature iterators (HuggingFace dataset → utt_id, feature array)
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_utt_ids(hf_ds) -> List[str]:
    """
    Extract stable utterance IDs by loading audio metadata with decode=False.

    This mirrors PretrainDataset exactly: Audio(decode=False) always returns a
    plain dict {"path": ..., "bytes": ...}, so path is reliably available even
    when the dataset comes from the HuggingFace Hub (where decoded rows have a
    None path after streaming).
    """
    try:
        from datasets import Audio
    except ImportError:
        return [f"utt_{i:06d}" for i in range(len(hf_ds))]

    path_ds  = hf_ds.cast_column("audio", Audio(sampling_rate=_TARGET_SR, decode=False))
    audio_meta = path_ds["audio"]
    ids = []
    for i, a in enumerate(audio_meta):
        path = (a.get("path") or "") if isinstance(a, dict) \
               else (getattr(a, "path", "") or "")
        stem = Path(path).stem
        ids.append(stem if stem else f"utt_{i:06d}")
    return ids


def _mfcc_iterator(
    hf_ds,
    utt_ids:    List[str],
    n_mfcc:     int,
    max_frames: Optional[int],
) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Yield (utt_id, mfcc_array) for every row in an HF dataset.
    The dataset must have its ``audio`` column cast to 16 kHz before calling.
    hop_length=320 aligns MFCC frames with HuBERT CNN frames.
    """
    transform = T.MFCC(
        sample_rate=_TARGET_SR,
        n_mfcc=n_mfcc,
        melkwargs=dict(
            n_fft=512,
            hop_length=_HUBERT_CNN_STRIDE,
            n_mels=80,
            center=False,
        ),
    )
    for i, row in enumerate(hf_ds):
        utt_id = utt_ids[i]
        try:
            arr = np.asarray(row["audio"]["array"], dtype=np.float32)
            wav = torch.from_numpy(arr)
            if wav.dim() == 2:
                wav = wav.mean(0)
            feat = transform(wav.unsqueeze(0)).squeeze(0).T.numpy()   # (T_feat, n_mfcc)
            if max_frames is not None:
                feat = feat[:max_frames]
            yield utt_id, feat
        except Exception as exc:
            log.warning("MFCC failed for %s (%s) — skipping.", utt_id, exc)


@torch.no_grad()
def _model_feature_iterator(
    model:      torch.nn.Module,
    hf_ds,
    utt_ids:    List[str],
    device:     torch.device,
    max_frames: Optional[int],
) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Yield (utt_id, feat_array) where feat is the f₁ output of MultiResHuBERT.
    """
    model.eval()
    for i, row in enumerate(hf_ds):
        utt_id = utt_ids[i]
        try:
            arr = np.asarray(row["audio"]["array"], dtype=np.float32)
            wav = torch.from_numpy(arr)
            if wav.dim() == 2:
                wav = wav.mean(0)
            wav = wav.unsqueeze(0).to(device)   # (1, T)

            cnn_out = model.feature_extractor(wav).transpose(1, 2)
            hidden = model.feature_projection(cnn_out)
            hidden = hidden + model.pos_conv_embed(hidden)
            hidden = model.encoder_ln(hidden)
            hidden = model.encoder_drop(hidden)
            for layer in model.f1_layers:
                hidden = layer(hidden)[0]

            feat = hidden.squeeze(0).cpu().numpy()   # (T_feat, H)
            if max_frames is not None:
                feat = feat[:max_frames]
            yield utt_id, feat
        except Exception as exc:
            log.warning("Model features failed for %s (%s) — skipping.", utt_id, exc)


# ─────────────────────────────────────────────────────────────────────────────
# K-means quantiser
# ─────────────────────────────────────────────────────────────────────────────

class KMeansQuantizer:
    """
    Wraps scikit-learn MiniBatchKMeans for incremental fitting on an HF dataset
    and provides label assignment + persistence.

    Args:
        n_clusters_hi: Number of clusters for the high-resolution head (g^q_R1).
        n_clusters_lo: Number of clusters for the low-resolution head (g^q_R2).
                       If None, hi labels are decimated instead of running a
                       separate k-means.
        kmeans_batch:  Mini-batch size for MiniBatchKMeans.partial_fit.
        n_init:        Number of k-means initialisations (best inertia kept).
        seed:          Random seed.
    """

    def __init__(
        self,
        n_clusters_hi: int = 100,
        n_clusters_lo: Optional[int] = None,
        kmeans_batch:  int  = 10_000,
        n_init:        int  = 3,
        seed:          int  = 42,
    ) -> None:
        from sklearn.cluster import MiniBatchKMeans

        self.n_clusters_hi = n_clusters_hi
        self.n_clusters_lo = n_clusters_lo
        self._kmeans_hi = MiniBatchKMeans(
            n_clusters   = n_clusters_hi,
            batch_size   = kmeans_batch,
            n_init       = n_init,
            random_state = seed,
            verbose      = 0,
        )
        self._kmeans_lo: Optional[object] = None
        if n_clusters_lo is not None:
            self._kmeans_lo = MiniBatchKMeans(
                n_clusters   = n_clusters_lo,
                batch_size   = kmeans_batch,
                n_init       = n_init,
                random_state = seed + 1,
                verbose      = 0,
            )
        self._fitted = False
        self._all_feats_cache: Dict[str, np.ndarray] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Incremental fitting (shared core)
    # ──────────────────────────────────────────────────────────────────────

    def _fit_incremental(
        self,
        feature_iter:      Iterator[Tuple[str, np.ndarray]],
        accumulate_frames: int = 500_000,
    ) -> None:
        """
        Stream features through MiniBatchKMeans.partial_fit in chunks.
        Caches all feature arrays for reuse in assign_and_save().
        """
        self._all_feats_cache = {}
        buf: List[np.ndarray] = []
        buf_len = 0

        log.info("Collecting features and fitting k-means incrementally …")
        for i, (utt_id, feat) in enumerate(feature_iter):
            self._all_feats_cache[utt_id] = feat
            buf.append(feat)
            buf_len += feat.shape[0]

            if buf_len >= accumulate_frames:
                chunk = np.concatenate(buf, axis=0)
                self._kmeans_hi.partial_fit(chunk)
                if self._kmeans_lo is not None:
                    self._kmeans_lo.partial_fit(chunk)
                buf, buf_len = [], 0
                log.debug("  partial_fit after %d utterances …", i + 1)

        if buf:
            chunk = np.concatenate(buf, axis=0)
            self._kmeans_hi.partial_fit(chunk)
            if self._kmeans_lo is not None:
                self._kmeans_lo.partial_fit(chunk)

        self._fitted = True
        log.info(
            "K-means fitted: hi=%d  lo=%s  (%d utterances)",
            self.n_clusters_hi,
            str(self.n_clusters_lo) if self.n_clusters_lo else "decimated",
            len(self._all_feats_cache),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public fit methods
    # ──────────────────────────────────────────────────────────────────────

    def fit_from_mfcc(
        self,
        hf_ds,
        n_mfcc:             int           = 39,
        max_frames_per_utt: Optional[int] = None,
        accumulate_frames:  int           = 500_000,
    ) -> None:
        """
        Fit k-means on MFCC features (iteration 1).

        Args:
            hf_ds: HuggingFace dataset with ``audio`` column cast to 16 kHz.
        """
        log.info("Pre-computing utterance IDs (decode=False) …")
        utt_ids = _precompute_utt_ids(hf_ds)
        log.info("  %d utterances  (first 3: %s)", len(utt_ids), utt_ids[:3])
        self._fit_incremental(
            _mfcc_iterator(hf_ds, utt_ids, n_mfcc, max_frames_per_utt),
            accumulate_frames,
        )

    def fit_from_model(
        self,
        model:              torch.nn.Module,
        hf_ds,
        device:             str           = "cuda" if torch.cuda.is_available() else "cpu",
        max_frames_per_utt: Optional[int] = None,
        accumulate_frames:  int           = 500_000,
    ) -> None:
        """
        Fit k-means on internal model features (iteration 2).

        Args:
            model: MultiResHuBERT in eval mode.
            hf_ds: HuggingFace dataset with ``audio`` column cast to 16 kHz.
        """
        log.info("Pre-computing utterance IDs (decode=False) …")
        utt_ids = _precompute_utt_ids(hf_ds)
        log.info("  %d utterances  (first 3: %s)", len(utt_ids), utt_ids[:3])
        dev = torch.device(device)
        model = model.to(dev)
        self._fit_incremental(
            _model_feature_iterator(model, hf_ds, utt_ids, dev, max_frames_per_utt),
            accumulate_frames,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Label assignment
    # ──────────────────────────────────────────────────────────────────────

    def _assign_hi(self, feat: np.ndarray) -> np.ndarray:
        return self._kmeans_hi.predict(feat).astype(np.int16)

    def _assign_lo(
        self,
        feat:              np.ndarray,
        hi_labels:         np.ndarray,
        downsample_stride: int,
    ) -> np.ndarray:
        if self._kmeans_lo is not None:
            return self._kmeans_lo.predict(feat).astype(np.int16)
        return hi_labels[::downsample_stride].astype(np.int16)

    def assign_and_save(
        self,
        output_path:       str,
        downsample_stride: int = 2,
    ) -> None:
        """
        Assign cluster labels to every utterance cached during fit and write
        ``{utt_id: {"hi": np.int16, "lo": np.int16}}`` to ``output_path``.

        Must be called after fit_from_mfcc() or fit_from_model().
        """
        if not self._fitted:
            raise RuntimeError("Call fit_from_mfcc() or fit_from_model() first.")
        if not self._all_feats_cache:
            raise RuntimeError("Feature cache is empty — re-run fit.")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        labels: Dict[str, Dict[str, np.ndarray]] = {}
        for utt_id, feat in self._all_feats_cache.items():
            hi = self._assign_hi(feat)
            lo = self._assign_lo(feat, hi, downsample_stride)
            labels[utt_id] = {"hi": hi, "lo": lo}

        with open(output_path, "wb") as f:
            pickle.dump(labels, f, protocol=4)
        log.info("Labels saved → %s  (%d utterances)", output_path, len(labels))

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_clusters_hi": self.n_clusters_hi,
            "n_clusters_lo": self.n_clusters_lo,
            "kmeans_hi":     self._kmeans_hi,
            "kmeans_lo":     self._kmeans_lo,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        log.info("KMeansQuantizer saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "KMeansQuantizer":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls.__new__(cls)
        obj.n_clusters_hi    = payload["n_clusters_hi"]
        obj.n_clusters_lo    = payload["n_clusters_lo"]
        obj._kmeans_hi       = payload["kmeans_hi"]
        obj._kmeans_lo       = payload["kmeans_lo"]
        obj._fitted          = True
        obj._all_feats_cache = {}
        log.info("KMeansQuantizer loaded from %s", path)
        return obj
