"""
Qwen3-Embedding text encoder for Fusion-C pronunciation assessment.

Wraps Qwen/Qwen3-Embedding-0.6B (decoder-only) as a sentence encoder that
can be either frozen or fine-tuned.  Output is an L2-normalised last-token
embedding (1024-dim), cast to float32.

Key differences from BERT-family (CLS-pooled) encoders:
  - Decoder-only architecture: last-token pooling, NOT [CLS]
  - Tokenizer padding_side MUST be "left" for correct batch pooling
  - Model runs in float16; output cast to float32
  - Instruct prefix improves embedding quality for this task
  - Requires transformers>=4.51.0

Standalone import:
    from model.text_encoder import Qwen3TextEncoder
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Qwen3TextEncoder(nn.Module):
    """
    Qwen3-Embedding-0.6B sentence encoder — frozen by default,
    optionally fine-tuneable.

    Args:
        model_name:  HuggingFace model ID
                     (default: ``"Qwen/Qwen3-Embedding-0.6B"``).
        max_length:  Tokenizer truncation length in tokens.
                     128 is sufficient for short pronunciation transcripts.
        frozen:      If True (default), parameters are frozen and forward runs
                     under ``torch.no_grad()``.  Set False to allow gradient
                     flow for fine-tuning; use a small lr (≤ 1e-5).

    Forward:
        transcripts: List[str]  — batch of raw transcript strings
        returns:     Tensor [B, 1024]  — L2-normalised last-token embeddings,
                                         float32
    """

    # Task-specific instruct prefix for pronunciation quality assessment.
    INSTRUCT_PREFIX: str = (
        "Instruct: Assess the pronunciation quality of the following "
        "spoken utterance transcription\nQuery: "
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        max_length: int = 128,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.51.0 is required for Qwen3TextEncoder.\n"
                "Install:  pip install 'transformers>=4.51.0'"
            ) from exc

        self.max_length = max_length
        # padding_side="left" is CRITICAL: with left padding, the last position
        # in every sequence is always a real token, making last-token pooling
        # correct and simple for variable-length batches.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )
        # float16 for memory efficiency; output is cast to float32 on return.
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self._frozen: bool = frozen

        if frozen:
            for param in self.model.parameters():
                param.requires_grad_(False)

    # ──────────────────────────────────────────────────────────────────────
    # Freeze / unfreeze
    # ──────────────────────────────────────────────────────────────────────

    def freeze(self) -> None:
        """Freeze all parameters; forward will run under no_grad."""
        self._frozen = True
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        """Unfreeze all parameters; gradients flow through forward."""
        self._frozen = False
        for p in self.parameters():
            p.requires_grad_(True)

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def output_dim(self) -> int:
        """Output embedding dimension (1024 for Qwen3-Embedding-0.6B)."""
        return 1024

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # ──────────────────────────────────────────────────────────────────────
    # Last-token pooling  (decoder-only requires this, NOT [CLS])
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def last_token_pool(
        last_hidden_state: Tensor,  # [B, T, H]
        attention_mask: Tensor,     # [B, T]  1=real 0=pad
    ) -> Tensor:
        """
        Extract the last real token's embedding for each sequence.

        Works for both left-padded (our case) and right-padded inputs.
        With left padding the final position is always a real token, so we
        can simply take last_hidden_state[:, -1].  With right padding we index
        per-sequence using the actual sequence lengths.
        """
        B = last_hidden_state.shape[0]
        left_padding = attention_mask[:, -1].sum() == B
        if left_padding:
            return last_hidden_state[:, -1]
        seq_lens = attention_mask.sum(dim=1) - 1   # 0-indexed last real token
        return last_hidden_state[
            torch.arange(B, device=last_hidden_state.device), seq_lens
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Internal encode  (shared by both forward paths)
    # ──────────────────────────────────────────────────────────────────────

    def _encode(self, transcripts: List[str]) -> Tensor:
        texts = [self.INSTRUCT_PREFIX + t for t in transcripts]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        output = self.model(**encoded)

        emb = self.last_token_pool(
            output.last_hidden_state, encoded["attention_mask"]
        )                                     # [B, 1024]  float16
        emb = F.normalize(emb, p=2, dim=1)   # L2 normalise
        return emb.float()                    # cast fp16 → fp32

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, transcripts: List[str]) -> Tensor:
        """
        Encode a batch of transcripts.

        The instruct prefix is prepended automatically before tokenization.

        Args:
            transcripts: List of B strings.

        Returns:
            Tensor [B, 1024] — L2-normalised last-token embeddings, float32.
            Gradient flows only when the encoder is unfrozen and the module
            is in training mode.
        """
        if not self.training or self._frozen:
            with torch.no_grad():
                return self._encode(transcripts)
        return self._encode(transcripts)    # grad flows when unfrozen + training
