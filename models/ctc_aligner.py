"""
CTC forced phoneme aligner — optional dashed path in the architecture diagram.

Uses torchaudio's MMS-FA (Massively Multilingual Speech Forced Alignment)
pipeline to produce phone-level timestamps from raw audio + a transcript.

These phone spans can be passed to HubertMultiTask.forward_inference() as
`phone_spans`, enabling the CTC path in the cross-attention fusion block:

    predictor.py:
        phone_spans = ctc_aligner.align(audio, sr, transcript, frame_hz)
        out = model.forward_inference(..., phone_spans=phone_spans)

Install torchaudio to enable this module:
    pip install torchaudio

Without torchaudio the module imports cleanly but `CTCPhoneAligner.available`
returns False and `align()` returns an empty list (no-op fallback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import torch
    import torchaudio                           # noqa: F401  (checked at runtime)
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PhoneSpan:
    """One phone segment produced by forced alignment."""
    phone:       str
    start_s:     float           # start time in seconds
    end_s:       float           # end time in seconds
    score:       float           # mean log-prob from forced_align
    frame_start: int = field(default=0)  # HuBERT frame index (at frame_hz)
    frame_end:   int = field(default=0)


# ---------------------------------------------------------------------------
# Aligner
# ---------------------------------------------------------------------------

class CTCPhoneAligner:
    """
    Forced phoneme aligner wrapping torchaudio MMS-FA.

    Parameters
    ----------
    device : 'cpu' or 'cuda'
    """

    def __init__(self, device: str = "cpu"):
        self._device  = device
        self._bundle  = None
        self._model   = None
        self._labels: List[str] = []
        self._dict:   dict      = {}

        if not _TORCHAUDIO_AVAILABLE:
            print("[CTCPhoneAligner] torchaudio not installed — phone alignment disabled.")
            return

        try:
            import torchaudio.pipelines as pip
            self._bundle = pip.MMS_FA
            self._model  = self._bundle.get_model(with_star=False).to(device)
            self._labels = list(self._bundle.get_labels(star=None))
            self._dict   = self._bundle.get_dict(star=None)
            print("[CTCPhoneAligner] torchaudio MMS-FA loaded successfully.")
        except Exception as exc:
            print(f"[CTCPhoneAligner] Could not load MMS-FA ({exc}) — phone alignment disabled.")
            self._bundle = None

    @property
    def available(self) -> bool:
        return self._bundle is not None

    # ------------------------------------------------------------------

    def align(
        self,
        audio:      np.ndarray,
        sr:         int,
        transcript: str,
        frame_hz:   float = 50.0,
    ) -> List[PhoneSpan]:
        """
        Forced-align `transcript` to `audio` and return phone-level spans.

        Parameters
        ----------
        audio      : float32 numpy array at `sr` Hz
        sr         : sample rate of `audio`
        transcript : space-separated word string (e.g. from ASR)
        frame_hz   : HuBERT frame rate used to compute frame_start/frame_end

        Returns
        -------
        List of PhoneSpan, empty list when aligner is unavailable or fails.
        """
        if not self.available:
            return []

        import torch
        import torchaudio
        from torchaudio.functional import forced_align

        waveform = torch.from_numpy(audio).float().unsqueeze(0).to(self._device)
        if sr != self._bundle.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self._bundle.sample_rate
            )

        with torch.inference_mode():
            emission, _ = self._model(waveform)  # (1, T_frames, vocab)

        # Build phone token sequence from transcript words
        words      = transcript.lower().strip().split()
        phone_seq  = []
        for w in words:
            phones = self._dict.get(w)
            if phones:
                phone_seq.extend(phones)
            else:
                # Fallback: treat each character as a "phone"
                phone_seq.extend(list(w))

        if not phone_seq:
            return []

        # Filter to tokens present in the label set
        valid_tokens = [(i, p) for i, p in enumerate(phone_seq) if p in self._labels]
        if not valid_tokens:
            return []

        idxs, phones_clean = zip(*valid_tokens)
        token_ids = torch.tensor(
            [[self._labels.index(p) for p in phones_clean]],
            dtype=torch.int32,
        ).to(self._device)

        try:
            frame_alignment, scores = forced_align(emission, token_ids, blank=0)
        except Exception as exc:
            print(f"[CTCPhoneAligner] forced_align failed: {exc}")
            return []

        merged   = _merge_tokens(
            frame_alignment[0].tolist(),
            scores[0].tolist(),
            list(phones_clean),
        )
        duration  = len(audio) / sr
        n_frames  = emission.size(1)

        spans: List[PhoneSpan] = []
        for phone, f_start, f_end, score in merged:
            start_s = f_start / n_frames * duration
            end_s   = f_end   / n_frames * duration
            spans.append(PhoneSpan(
                phone       = phone,
                start_s     = start_s,
                end_s       = end_s,
                score       = float(score),
                frame_start = int(start_s * frame_hz),
                frame_end   = int(end_s   * frame_hz),
            ))

        return spans


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _merge_tokens(
    alignment: List[int],
    scores:    List[float],
    tokens:    List[str],
) -> List[tuple]:
    """
    Collapse a frame-level token alignment into (token, f_start, f_end, mean_score).

    Blank frames (index 0) are skipped; consecutive repeated token frames are
    merged into a single span.
    """
    merged:  List[tuple] = []
    tok_idx: int         = 0
    i:       int         = 0
    n:       int         = len(alignment)

    while i < n and tok_idx < len(tokens):
        val = alignment[i]
        if val == 0:        # CTC blank — skip
            i += 1
            continue
        j = i
        while j < n and alignment[j] == val:
            j += 1
        mean_score = float(np.mean(scores[i:j]))
        merged.append((tokens[tok_idx], i, j, mean_score))
        tok_idx += 1
        i = j

    return merged
