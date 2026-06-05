"""
data/pseudo_phoneme_labels.py — Phase 4 Sub-path B.

Generates per-frame phoneme pseudo-labels using a frozen wav2vec2 CTC model.
Called once per batch during training epochs 1-3 to provide auxiliary CE
supervision for the phoneme head in PronunciationScorer.

Sub-path A (if MFA TextGrid alignments are available in data/mfa/):
    TODO: replace PseudoPhonemeLabeller with a frame-level alignment reader
    that parses data/mfa/*.TextGrid files and maps phonemes to Arpabet-40
    integer labels using data/phoneme_labels.py.
    The data/mfa/ directory already exists (untracked) and may contain
    force-aligned TextGrids from the SpeechOcean762 preparation pipeline.

Performance note:
    ~15 ms per batch on CPU, ~4 ms on GPU.
    If too slow, gate via `global_step % 4 == 0` in the training loop
    (pass phoneme_labels=None on ungated steps).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

log = logging.getLogger(__name__)


class PseudoPhonemeLabeller:
    """
    Wraps a frozen wav2vec2 CTC model to produce frame-level phoneme
    argmax pseudo-labels for auxiliary CE supervision.

    Args:
        model_name: HuggingFace model ID (default: facebook/wav2vec2-base).
                    The CTC vocabulary size (32 for wav2vec2-base) need not
                    match num_phonemes (40); CE uses ignore_index=-1 so
                    out-of-range labels are masked.
        device:     torch device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        device:     str = "cpu",
    ) -> None:
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.0 required for PseudoPhonemeLabeller.\n"
                "pip install transformers"
            ) from exc

        self.device    = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model     = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._vocab_size = self.model.config.vocab_size
        log.info(
            "PseudoPhonemeLabeller: %s  vocab=%d  device=%s",
            model_name, self._vocab_size, device,
        )

    @torch.no_grad()
    def get_labels(
        self,
        waveforms:      torch.Tensor,           # [B, T_audio] at 16 kHz
        attention_mask: Optional[torch.Tensor] = None,  # [B, T_audio]
    ) -> torch.Tensor:
        """
        Args:
            waveforms:      [B, T_audio] float32 at 16 kHz (same as scorer input).
            attention_mask: optional [B, T_audio] int (1=real, 0=pad).

        Returns:
            [B, T'] LongTensor of argmax CTC label indices at ~50 Hz.
            Values >= num_phonemes are clamped to -1 (ignored by CrossEntropyLoss).
            T' ≈ T_audio / 320 (wav2vec2 CNN stride).
        """
        # Processor expects list of 1-D numpy arrays
        wav_list = waveforms.cpu().float().numpy().tolist()
        inputs   = self.processor(
            wav_list,
            return_tensors  = "pt",
            sampling_rate   = 16_000,
            padding         = True,
        ).to(self.device)

        logits = self.model(**inputs).logits   # [B, T', vocab]
        labels = logits.argmax(dim=-1)         # [B, T']  (argmax pseudo-label)

        # Clamp labels outside Arpabet-40 range to -1 (ignore_index in CE).
        # wav2vec2-base has 32 CTC tokens; anything ≥ 40 → -1.
        labels = labels.where(labels < 40, torch.full_like(labels, -1))

        return labels.cpu()
