"""
Qwen3-Embedding text encoder for the pronunciation scoring pipeline.

Identical to model/text_encoder.py (Qwen3TextEncoder) except:
  - Uses mask-aware MEAN pooling instead of last-token pooling.
    This matches the diagram annotation and is more robust for
    variable-length sequences when the model is not strictly causal.
  - Class is named Qwen3MeanPoolEncoder to avoid import collisions.

Key properties (same as Qwen3TextEncoder):
  - Decoder-only: padding_side="left" is critical
  - Runs in float16; output cast to float32
  - Instruct prefix for pronunciation task
  - Requires transformers>=4.51.0
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Qwen3MeanPoolEncoder(nn.Module):
    """
    Qwen3-Embedding-0.6B with mask-aware mean pooling.

    Args:
        model_name:  HuggingFace model ID (default: ``"Qwen/Qwen3-Embedding-0.6B"``).
        max_length:  Tokenizer truncation length (tokens).
        frozen:      If True (default), all parameters are frozen; forward runs
                     under ``torch.no_grad()``.

    Forward:
        transcripts: List[str]  — batch of transcripts
        returns:     Tensor [B, 1024]  — L2-normalised mean-pool embeddings,
                                         float32
    """

    INSTRUCT_PREFIX: str = (
        "Instruct: Assess the pronunciation quality of the following "
        "spoken utterance transcription\nQuery: "
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        max_length: int = 128,
        frozen: bool = True,
        padding_side: str = "left",
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.51.0 is required for Qwen3MeanPoolEncoder.\n"
                "Install:  pip install 'transformers>=4.51.0'"
            ) from exc

        self.max_length = max_length
        self.tokenizer  = AutoTokenizer.from_pretrained(
            model_name, padding_side=padding_side
        )
        self.model  = AutoModel.from_pretrained(
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
        self._frozen = True
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        self._frozen = False
        for p in self.parameters():
            p.requires_grad_(True)

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def output_dim(self) -> int:
        return 1024

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # ──────────────────────────────────────────────────────────────────────
    # Mask-aware mean pooling  (diagram: "mask-aware mean pool → (1024,)")
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def mask_aware_mean_pool(
        last_hidden_state: Tensor,   # [B, T, H]
        attention_mask: Tensor,      # [B, T]  1=real 0=pad
    ) -> Tensor:
        """
        Weighted mean over real token positions, L2-normalised.

        Returns:
            [B, H]
        """
        mask   = attention_mask.unsqueeze(-1).float()          # [B, T, 1]
        pooled = (last_hidden_state.float() * mask).sum(1) / \
                 mask.sum(1).clamp(min=1e-9)                   # [B, H]
        return F.normalize(pooled, p=2, dim=1)

    # ──────────────────────────────────────────────────────────────────────
    # Internal encode
    # ──────────────────────────────────────────────────────────────────────

    def _encode(self, transcripts: List[str]) -> Tensor:
        texts   = [self.INSTRUCT_PREFIX + t for t in transcripts]
        encoded = self.tokenizer(
            texts,
            padding    = True,
            truncation = True,
            max_length = self.max_length,
            return_tensors = "pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        output  = self.model(**encoded)

        emb = self.mask_aware_mean_pool(
            output.last_hidden_state,
            encoded["attention_mask"],
        )                          # [B, 1024]  float32, L2-normalised
        return emb

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, transcripts: List[str]) -> Tensor:
        """
        Encode a batch of transcripts with instruct prefix.

        Returns:
            Tensor [B, 1024] — L2-normalised mean-pool embeddings, float32.
            Gradient flows only when unfrozen and in training mode.
        """
        if not self.training or self._frozen:
            with torch.no_grad():
                return self._encode(transcripts)
        return self._encode(transcripts)
