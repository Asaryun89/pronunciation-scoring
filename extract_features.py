#!/usr/bin/env python3
"""
Extract intermediate representations from a Multi-resolution HuBERT model.

Each audio file produces four .npy arrays:

  h0  (T_feat, H_cnn)   — CNN feature extractor output (after projection)
  h1  (T_feat, H)       — output of f₁ (high-res encoder, before DOWN)
  h2  (T_feat', H)      — output of f₂ (low-res encoder, after DOWN)
  h3  (T_feat, H)       — output of f₃ (final high-res encoder)

where T_feat ≈ T_audio / 320 and T_feat' = T_feat // downsample_stride.

These representations are used for:
  • Probing classifiers (phoneme, prosody, speaker)
  • K-means target generation (see kmeans_clustering.py)
  • Visualisation / t-SNE analysis

Usage
─────
    # Single file
    python extract_features.py \\
        --config     configs/finetune.yaml \\
        --checkpoint checkpoints/finetune/best_model.pt \\
        --audio      path/to/audio.wav \\
        --output-dir features/

    # All utterances in a Kaldi wav.scp
    python extract_features.py \\
        --config     configs/finetune.yaml \\
        --checkpoint checkpoints/finetune/best_model.pt \\
        --wav-scp    data/SPEECHOCEAN762/test/wav.scp \\
        --output-dir features/test/ \\
        --layers     h1,h2,h3      # comma-separated, default: h0,h1,h2,h3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import yaml
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent))

from model import MultiResHuBERT
from model.multi_res_hubert import _feat_mask_to_additive, _run_transformer_layers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_TARGET_SR = 16_000


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio(path: str, max_seconds: float = 60.0) -> Tensor:
    """Load, mono-mix, resample to 16 kHz.  Returns (T,) tensor."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != _TARGET_SR:
        wav = T.Resample(sr, _TARGET_SR)(wav)
    max_samples = int(max_seconds * _TARGET_SR)
    wav = wav[:, :max_samples]
    return wav.squeeze(0)   # (T,)


def _parse_wav_scp(path: str) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                entries[parts[0]] = parts[1]
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (step-by-step forward with hooks for h1)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_representations(
    model:     MultiResHuBERT,
    waveform:  Tensor,                   # (T,) on CPU
    device:    torch.device,
    layers:    Set[str],                 # which layers to return
    max_seconds: float = 60.0,
) -> Dict[str, np.ndarray]:
    """
    Run a single waveform through the model and return intermediate
    feature maps as numpy arrays.

    Args:
        model:      Loaded MultiResHuBERT in eval mode.
        waveform:   1-D float32 tensor (T,) at 16 kHz.
        device:     Target device.
        layers:     Set of layer names to return: {"h0", "h1", "h2", "h3"}.

    Returns:
        Dict mapping layer name → (T, H) float32 numpy array.
    """
    wav = waveform.unsqueeze(0).to(device)   # (1, T)

    # ── f₀: CNN feature extraction ────────────────────────────────────────
    cnn_out = model.feature_extractor(wav).transpose(1, 2)   # (1, T_feat, H_cnn)
    hidden = model.feature_projection(cnn_out)                 # (1, T_feat, H)

    results: Dict[str, np.ndarray] = {}
    if "h0" in layers:
        results["h0"] = hidden.squeeze(0).cpu().float().numpy()   # (T_feat, H)

    # ── Positional encoding (once, before f₁) ─────────────────────────────
    hidden = hidden + model.pos_conv_embed(hidden)
    hidden = model.encoder_ln(hidden)
    hidden = model.encoder_drop(hidden)

    # ── f₁: High-resolution encoder ──────────────────────────────────────
    h1 = _run_transformer_layers(model.f1_layers, hidden, attn_mask=None)

    if "h1" in layers:
        results["h1"] = h1.squeeze(0).cpu().float().numpy()   # (T_feat, H)

    # ── DOWN: temporal downsampling ───────────────────────────────────────
    h1_down, _ = model.down(h1)   # (1, T_feat', H)

    # ── f₂: Low-resolution encoder ───────────────────────────────────────
    h2 = _run_transformer_layers(model.f2_layers, h1_down, attn_mask=None)

    if "h2" in layers:
        results["h2"] = h2.squeeze(0).cpu().float().numpy()   # (T_feat', H)

    # ── UP: upsample + skip ───────────────────────────────────────────────
    h2_up = model.up(h2, h1)   # (1, T_feat, H)

    # ── f₃: High-resolution encoder ──────────────────────────────────────
    h3 = _run_transformer_layers(model.f3_layers, h2_up, attn_mask=None)

    if "h3" in layers:
        results["h3"] = h3.squeeze(0).cpu().float().numpy()   # (T_feat, H)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_wav_scp(
    model:      MultiResHuBERT,
    wav_scp:    Dict[str, str],
    output_dir: Path,
    device:     torch.device,
    layers:     Set[str],
    max_seconds: float = 60.0,
    skip_existing: bool = True,
) -> None:
    """
    Extract and save features for every utterance in ``wav_scp``.

    Output layout::

        output_dir/
          <utt_id>.h0.npy
          <utt_id>.h1.npy
          <utt_id>.h2.npy
          <utt_id>.h3.npy
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(wav_scp)
    n_done  = 0

    for utt_id, wav_path in wav_scp.items():
        # Skip if all requested layers already extracted
        target_files = [output_dir / f"{utt_id}.{l}.npy" for l in layers]
        if skip_existing and all(p.exists() for p in target_files):
            n_done += 1
            continue

        try:
            wav     = _load_audio(wav_path, max_seconds)
            feats   = extract_representations(model, wav, device, layers, max_seconds)

            for layer_name, arr in feats.items():
                np.save(output_dir / f"{utt_id}.{layer_name}.npy", arr)

            n_done += 1
            if n_done % 100 == 0 or n_done == n_total:
                log.info("  %d / %d utterances processed", n_done, n_total)

        except Exception as exc:
            log.warning("Failed for %s (%s) — skipping.", utt_id, exc)

    log.info("Feature extraction complete.  %d utterances → %s", n_done, output_dir)


def process_single_file(
    model:      MultiResHuBERT,
    audio_path: str,
    output_dir: Path,
    device:     torch.device,
    layers:     Set[str],
    max_seconds: float = 60.0,
) -> None:
    """Extract and save features for a single audio file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem

    wav   = _load_audio(audio_path, max_seconds)
    feats = extract_representations(model, wav, device, layers, max_seconds)

    for layer_name, arr in feats.items():
        out_path = output_dir / f"{stem}.{layer_name}.npy"
        np.save(out_path, arr)
        log.info("  saved %s  shape=%s", out_path.name, arr.shape)


# ─────────────────────────────────────────────────────────────────────────────
# Concatenate utility (for downstream k-means)
# ─────────────────────────────────────────────────────────────────────────────

def concatenate_layer(
    features_dir: Path,
    layer:        str,
    max_utts:     Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load all ``*.{layer}.npy`` files from ``features_dir``, concatenate frames,
    and return (frames_array, list_of_utt_ids).

    Useful for fitting k-means on extracted representations.

    Args:
        features_dir: Directory produced by process_wav_scp.
        layer:        One of "h0", "h1", "h2", "h3".
        max_utts:     If set, only load the first N utterances.

    Returns:
        frames:  (N_total_frames, H)
        utt_ids: list of utterance identifiers (one per file)
    """
    files = sorted(features_dir.glob(f"*.{layer}.npy"))
    if max_utts is not None:
        files = files[:max_utts]

    all_frames: List[np.ndarray] = []
    utt_ids:    List[str]        = []

    for p in files:
        arr = np.load(p)   # (T, H)
        all_frames.append(arr)
        utt_ids.append(p.stem.replace(f".{layer}", ""))

    return np.concatenate(all_frames, axis=0), utt_ids


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MultiResHuBERT features")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio",   help="Single audio file path")
    source.add_argument("--wav-scp", help="Kaldi wav.scp file")

    parser.add_argument("--config",      required=True, help="YAML config")
    parser.add_argument("--checkpoint",  required=True, help="Model checkpoint .pt")
    parser.add_argument("--output-dir",  default="features/")
    parser.add_argument("--layers",      default="h0,h1,h2,h3",
                        help="Comma-separated list of layers to extract")
    parser.add_argument("--max-seconds", type=float, default=60.0,
                        help="Truncate audio longer than this (seconds)")
    parser.add_argument("--device",      default=None)
    parser.add_argument("--no-skip",     action="store_true",
                        help="Re-extract even if output file already exists")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device  = torch.device(args.device or cfg["training"].get("device", "cpu"))
    layers  = set(x.strip() for x in args.layers.split(","))
    out_dir = Path(args.output_dir)

    valid_layers = {"h0", "h1", "h2", "h3"}
    unknown = layers - valid_layers
    if unknown:
        parser.error(f"Unknown layers: {unknown}.  Valid: {valid_layers}")

    # ── load model ────────────────────────────────────────────────────────
    log.info("Loading model from %s …", args.checkpoint)
    model = MultiResHuBERT(**cfg["model"])
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    log.info("Extracting layers: %s", sorted(layers))

    # ── extract ───────────────────────────────────────────────────────────
    if args.audio:
        process_single_file(model, args.audio, out_dir, device,
                            layers, args.max_seconds)
    else:
        wav_scp = _parse_wav_scp(args.wav_scp)
        process_wav_scp(model, wav_scp, out_dir, device,
                        layers, args.max_seconds,
                        skip_existing=not args.no_skip)


if __name__ == "__main__":
    main()
