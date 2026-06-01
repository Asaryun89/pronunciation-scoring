"""
Speechocean762 dataset with Faster-Whisper ASR transcripts.

Two-phase workflow
──────────────────
Phase 1  (run ONCE, offline):
    python -m data.speechocean_asr --config configs/scoring_config.yaml
    OR
    from data.speechocean_asr import cache_asr_transcripts
    cache_asr_transcripts(cfg)

    Runs Faster-Whisper on every utterance and saves
    {utt_id: transcript_str} to cfg["data"]["asr_cache_path"].

Phase 2  (during training):
    dataset = SpeechOceanASRDataset(cfg, split="train")
    loader  = DataLoader(dataset, collate_fn=asr_collate_fn, ...)

Audio preprocessing per sample
───────────────────────────────
    1. Load from HF dataset at native SR
    2. Mono-mix if stereo
    3. Resample to target_sr (16 kHz)
    4. VAD trim (silero-vad; falls back to no-trim if unavailable)
    5. Normalise:  wav /= (|wav|.max() + 1e-8)

Score order in labels tensor
─────────────────────────────
    [total, accuracy, fluency, prosodic, completeness]  — normalised to [0, 1]
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

SCORE_KEYS: List[str] = ["total", "accuracy", "fluency", "prosodic", "completeness"]
SCORE_MAX:  float     = 10.0
TARGET_SR:  int       = 16_000

# HF dataset field name candidates for the "total" score column.
_TOTAL_CANDIDATES: List[str] = ["score", "total", "total_score"]
_COMPLETENESS_CANDIDATES: List[str] = ["completeness"]


# ─────────────────────────────────────────────────────────────────────────────
# VAD trim (silero-vad; graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

_vad_model = None


def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        try:
            from silero_vad import load_silero_vad
            _vad_model = load_silero_vad()
            log.debug("Silero VAD model loaded.")
        except Exception as exc:
            log.warning("silero-vad unavailable (%s) — VAD trim skipped.", exc)
            _vad_model = False   # sentinel: unavailable
    return _vad_model


def _vad_trim(wav: Tensor, sr: int, threshold: float = 0.5) -> Tensor:
    """Trim leading/trailing silence with silero-vad.  No-op if unavailable."""
    model = _get_vad_model()
    if not model:
        return wav
    try:
        from silero_vad import get_speech_timestamps
        ts = get_speech_timestamps(wav, model, sampling_rate=sr, threshold=threshold)
        if ts:
            return wav[ts[0]["start"]: ts[-1]["end"]]
    except Exception:
        pass
    return wav


# ─────────────────────────────────────────────────────────────────────────────
# Audio preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_wav(
    audio_array:   np.ndarray,
    src_sr:        int,
    target_sr:     int   = TARGET_SR,
    vad_threshold: float = 0.5,
    max_samples:   Optional[int] = None,
) -> Tensor:
    """
    Resample → mono → VAD trim → normalise.

    Args:
        audio_array:   Raw audio array from HF dataset.
        src_sr:        Source sample rate.
        target_sr:     Target sample rate (default 16 kHz).
        vad_threshold: Silero VAD probability threshold.
        max_samples:   Truncate to this many samples after resampling.

    Returns:
        Tensor [T] float32, normalised to [-1, 1].
    """
    import torchaudio

    wav = torch.from_numpy(np.asarray(audio_array, dtype=np.float32))
    if wav.dim() == 2:
        wav = wav.mean(0)

    if src_sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=src_sr, new_freq=target_sr)

    wav = _vad_trim(wav, target_sr, threshold=vad_threshold)

    if max_samples is not None:
        wav = wav[:max_samples]

    wav = wav / (wav.abs().max() + 1e-8)
    return wav


# ─────────────────────────────────────────────────────────────────────────────
# HF dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_col(column_names: List[str], candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in column_names:
            return c
    raise ValueError(
        f"Cannot find '{label}' column.  Tried {candidates}.  "
        f"Available: {sorted(column_names)}"
    )


def _as_audio_dict(audio) -> dict:
    if isinstance(audio, dict):
        return audio
    try:
        path = audio["path"]
    except (KeyError, TypeError):
        path = None
    return {
        "array":         audio["array"],
        "sampling_rate": audio["sampling_rate"],
        "path":          path,
    }


def _utt_id(audio_dict: dict, idx: int) -> str:
    p = audio_dict.get("path") or ""
    return Path(p).stem if p else f"utt_{idx:06d}"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — offline ASR transcript caching
# ─────────────────────────────────────────────────────────────────────────────

def cache_asr_transcripts(cfg: dict) -> None:
    """
    Run Faster-Whisper on every Speechocean762 utterance and cache results.

    Args:
        cfg: Full config dict.  Reads data.* keys.

    Outputs:
        ``cfg["data"]["asr_cache_path"]``:  pickle file containing
        ``{utt_id: transcript_str}`` for all train + test utterances.

    Skips silently if the cache file already exists.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is required for ASR caching.\n"
            "Install:  pip install faster-whisper"
        ) from exc

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc

    from tqdm import tqdm

    dcfg       = cfg["data"]
    cache_path = Path(dcfg["asr_cache_path"])

    if cache_path.exists():
        log.info("ASR cache already exists at %s — skipping.", cache_path)
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading Faster-Whisper model: %s …", dcfg["asr_model"])
    model = WhisperModel(
        dcfg["asr_model"],
        device       = dcfg.get("asr_device", "cpu"),
        compute_type = dcfg.get("asr_compute_type", "float16"),
    )

    hf_id  = dcfg.get("hf_dataset_id", "mispeech/speechocean762")
    splits = [dcfg.get("train_split", "train"), dcfg.get("test_split", "test")]
    cache: Dict[str, str] = {}

    for split in splits:
        log.info("Transcribing split=%s …", split)
        ds = load_dataset(hf_id, split=split)

        for idx, row in enumerate(tqdm(ds, desc=split)):
            audio  = _as_audio_dict(row["audio"])
            utt_id = _utt_id(audio, idx)

            # Transcribe in-memory (no disk write per sample)
            wav = preprocess_wav(
                audio["array"], audio["sampling_rate"],
                target_sr     = dcfg.get("target_sr", TARGET_SR),
                vad_threshold = dcfg.get("vad_threshold", 0.5),
            )
            # Faster-Whisper expects numpy float32
            segments, _ = model.transcribe(wav.numpy(), language="en")
            transcript  = " ".join(s.text.strip() for s in segments).strip()
            cache[utt_id] = transcript

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    log.info("ASR cache saved: %s  (%d utterances)", cache_path, len(cache))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — training dataset
# ─────────────────────────────────────────────────────────────────────────────

class SpeechOceanASRDataset(Dataset):
    """
    HuggingFace-backed Speechocean762 dataset using ASR-generated transcripts.

    Each sample returns a dict with keys:
        utt_id:     str
        waveform:   Tensor [T]    — preprocessed 16 kHz audio
        length:     Tensor (long) — number of valid samples
        labels:     Tensor [5]    — [total, acc, flu, pro, comp] in [0, 1]
        transcript: str           — ASR-generated transcript

    Args:
        cfg:     Full config dict.
        split:   ``"train"`` or ``"test"``.
        augment: Add small Gaussian noise (currently unused; reserved).
    """

    def __init__(
        self,
        cfg:     dict,
        split:   str  = "train",
        augment: bool = False,
    ) -> None:
        super().__init__()
        try:
            from datasets import load_dataset, Audio as HFAudio
        except ImportError as exc:
            raise ImportError("pip install datasets") from exc

        dcfg = cfg["data"]
        self.max_samples   = int(dcfg.get("max_duration_s", 20.0) * TARGET_SR)
        self.vad_threshold = dcfg.get("vad_threshold", 0.5)
        self.target_sr     = dcfg.get("target_sr", TARGET_SR)
        self.augment       = augment

        # Load HF dataset and cast audio column.
        hf_id   = dcfg.get("hf_dataset_id", "mispeech/speechocean762")
        hf_ds   = load_dataset(hf_id, split=split,
                               cache_dir=dcfg.get("hf_cache_dir", None))
        hf_ds   = hf_ds.cast_column("audio", HFAudio(sampling_rate=self.target_sr))
        self._ds = hf_ds

        # Resolve score field names.
        cols            = hf_ds.column_names
        total_col       = _resolve_col(cols, _TOTAL_CANDIDATES, "total")
        complete_col    = _resolve_col(cols, _COMPLETENESS_CANDIDATES, "completeness")

        # Build (N, 5) label tensor: [total, acc, flu, pro, comp] in [0, 1]
        label_cols = [
            np.array(hf_ds[total_col],       dtype=np.float32) / SCORE_MAX,
            np.array(hf_ds["accuracy"],       dtype=np.float32) / SCORE_MAX,
            np.array(hf_ds["fluency"],        dtype=np.float32) / SCORE_MAX,
            np.array(hf_ds["prosodic"],       dtype=np.float32) / SCORE_MAX,
            np.array(hf_ds[complete_col],     dtype=np.float32) / SCORE_MAX,
        ]
        self.labels: Tensor = torch.from_numpy(
            np.column_stack(label_cols)
        )                              # (N, 5)

        # Load ASR cache.
        cache_path = Path(dcfg["asr_cache_path"])
        if not cache_path.exists():
            raise FileNotFoundError(
                f"ASR cache not found: {cache_path}\n"
                f"Run:  python -m data.speechocean_asr --config <config>"
            )
        with open(cache_path, "rb") as f:
            self._asr: Dict[str, str] = pickle.load(f)

        log.info(
            "SpeechOceanASRDataset [%s]: %d utterances, %d in ASR cache",
            split, len(hf_ds), len(self._asr),
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        row   = self._ds[idx]
        audio = _as_audio_dict(row["audio"])

        waveform = preprocess_wav(
            audio["array"],
            audio["sampling_rate"],
            target_sr     = self.target_sr,
            vad_threshold = self.vad_threshold,
            max_samples   = self.max_samples,
        )

        utt_id     = _utt_id(audio, idx)
        transcript = self._asr.get(utt_id, "")   # empty string if not in cache

        return {
            "utt_id":     utt_id,
            "waveform":   waveform,                       # [T] float32
            "length":     torch.tensor(waveform.shape[0], dtype=torch.long),
            "labels":     self.labels[idx],               # [5] float32
            "transcript": transcript,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────

def asr_collate_fn(
    batch: List[dict],
) -> dict:
    """
    Pad waveforms to the longest in the batch.

    Returns dict with keys:
        waveforms:      [B, T_max]   float32  — zero-padded
        attention_mask: [B, T_max]   long     — 1=real, 0=pad
        lengths:        [B]          long
        labels:         [B, 5]       float32  — [0, 1] normalised scores
        transcripts:    List[str]
        utt_ids:        List[str]
    """
    waveforms:   List[Tensor] = []
    lengths:     List[Tensor] = []
    labels:      List[Tensor] = []
    transcripts: List[str]    = []
    utt_ids:     List[str]    = []

    for s in batch:
        waveforms.append(s["waveform"])
        lengths.append(s["length"])
        labels.append(s["labels"])
        transcripts.append(s["transcript"])
        utt_ids.append(s["utt_id"])

    max_len = max(w.shape[0] for w in waveforms)
    padded  = torch.stack(
        [F.pad(w, (0, max_len - w.shape[0])) for w in waveforms]
    )                                              # [B, T_max]

    length_t = torch.stack(lengths)               # [B]
    attn     = (
        torch.arange(max_len).unsqueeze(0) < length_t.unsqueeze(1)
    ).long()                                      # [B, T_max]  1=real 0=pad

    return {
        "waveforms":      padded,
        "attention_mask": attn,
        "lengths":        length_t,
        "labels":         torch.stack(labels),    # [B, 5]
        "transcripts":    transcripts,
        "utt_ids":        utt_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point — run this to build the ASR cache
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Cache Faster-Whisper ASR transcripts for Speechocean762."
    )
    parser.add_argument(
        "--config", default="configs/scoring_config.yaml",
        help="Path to scoring_config.yaml"
    )
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cache_asr_transcripts(cfg)


if __name__ == "__main__":
    _main()
