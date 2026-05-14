from __future__ import annotations

"""
BGETextEmbedder: lightweight text branch for pronunciation scoring.

Wraps BAAI/bge-small-en-v1.5 and projects sentence embeddings to d_model
for cross-attention fusion with HuBERT audio features.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class BGETextEmbedder(nn.Module):
    """Sentence embedder built on BAAI/bge-small-en-v1.5.

    Encodes a tokenised script and returns a single sentence vector per
    example, shaped ``(B, 1, d_model)``, ready for use as Key/Value in
    :class:`~models.scoring_heads.CrossAttentionFusion`.

    Why mean pooling (not CLS):
        BGE is trained with mean pooling as its pooling strategy — the CLS
        token is not specially trained for sentence representation.

    Why L=1 in output:
        The original paper conditions audio attention on a single
        sentence-level vector, not per-token representations.  This keeps
        the cross-attention lightweight and avoids padding issues.

    Why freeze_encoder=True by default:
        BGE already provides strong sentence representations.  Fine-tuning
        on the small SpeechOcean762 dataset (5000 utterances) risks
        catastrophic forgetting of generalised text semantics.

    Args:
        d_model: Output embedding dimension.
        model_name: HuggingFace model ID for the BGE encoder.
        freeze_encoder: If True, all BGE encoder parameters are frozen.
        dropout: Dropout probability applied after LayerNorm.
    """

    def __init__(
        self,
        d_model: int = 256,
        model_name: str = "BAAI/bge-small-en-v1.5",
        freeze_encoder: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # BGE-small hidden size = 384
        self.proj = nn.Linear(384, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked mean pooling over token dimension.

        Args:
            last_hidden_state: ``(B, L, H)`` encoder output.
            attention_mask: ``(B, L)`` 1=valid, 0=pad.

        Returns:
            ``(B, H)`` sentence embedding.
        """
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokenised script text into a sentence embedding.

        Args:
            input_ids: ``(B, L)`` tokenised script.
            attention_mask: ``(B, L)`` 1=valid, 0=pad.

        Returns:
            ``(B, 1, d_model)`` sentence embedding with L=1.
        """
        out = self.encoder(input_ids, attention_mask)
        text_emb = self._mean_pool(out.last_hidden_state, attention_mask)  # (B, 384)
        text_emb = self.proj(text_emb)                                      # (B, d_model)
        text_emb = self.dropout(self.norm(text_emb))
        return text_emb.unsqueeze(1)                                        # (B, 1, d_model)
