"""
BGE text encoder for Fusion-C pronunciation assessment.

Wraps BAAI/bge-small-en-v1.5 (or any compatible BGE model) as a *frozen*
sentence encoder.  The output is the L2-normalised [CLS] token embedding,
which BGE is designed to use for cosine-similarity tasks.

Standalone import:
    from model.text_encoder import BGETextEncoder
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BGETextEncoder(nn.Module):
    """
    Frozen BGE sentence encoder.

    Args:
        model_name:  HuggingFace model ID (e.g. ``"BAAI/bge-small-en-v1.5"``).
        max_length:  Tokenizer truncation length (tokens).

    Forward:
        transcripts: List[str]  — batch of raw transcript strings
        returns:     Tensor [B, text_dim]  — L2-normalised [CLS] embeddings
    """

    # BGE requires this prefix for best-quality sentence embeddings.
    PREFIX: str = "Represent this sentence: "

    def __init__(self, model_name: str, max_length: int = 128) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for BGETextEncoder.\n"
                "Install:  pip install transformers"
            ) from exc

        self.max_length = max_length
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.model      = AutoModel.from_pretrained(model_name)

        # Freeze ALL weights — text encoder is a fixed feature extractor.
        for param in self.model.parameters():
            param.requires_grad_(False)

    # ──────────────────────────────────────────────────────────────────────
    # Device helper
    # ──────────────────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, transcripts: List[str]) -> Tensor:
        """
        Encode a batch of transcripts.

        Args:
            transcripts: List of B strings.

        Returns:
            Tensor [B, text_dim] — L2-normalised [CLS] embeddings on the same
            device as the encoder weights.
        """
        # BGE requires the task prefix for best embedding quality.
        texts = [self.PREFIX + t for t in transcripts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Move tokenised inputs to the model's device.
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        output  = self.model(**encoded)
        cls_emb = output.last_hidden_state[:, 0, :]   # [B, text_dim]

        return F.normalize(cls_emb.float(), p=2, dim=-1)
