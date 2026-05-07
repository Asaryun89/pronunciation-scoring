from __future__ import annotations

"""
PhonemeEmbedder: learned phoneme token + positional embeddings for cross-attention.
"""

import torch
import torch.nn as nn

from models.phoneme_vocab import PAD_ID, VOCAB_SIZE


class PhonemeEmbedder(nn.Module):
    """Encode a sequence of phoneme IDs into dense embeddings.

    Combines learned token embeddings with learned positional encodings, then
    applies a linear projection followed by layer normalisation and dropout.
    Sequences longer than ``max_len`` are silently truncated.

    Args:
        d_model: Output embedding dimension.
        vocab_size: Phoneme vocabulary size (default: :data:`~models.phoneme_vocab.VOCAB_SIZE`).
        max_len: Maximum sequence length; longer sequences are truncated.
        dropout: Dropout probability applied after normalisation.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, phoneme_ids: torch.Tensor) -> torch.Tensor:
        """Embed a batch of phoneme ID sequences.

        Args:
            phoneme_ids: ``(B, P)`` integer tensor of phoneme vocabulary IDs.

        Returns:
            ``(B, P, d_model)`` float tensor of projected phoneme embeddings.
        """
        if phoneme_ids.size(1) > self.max_len:
            phoneme_ids = phoneme_ids[:, : self.max_len]

        B, P = phoneme_ids.shape
        positions = torch.arange(P, device=phoneme_ids.device).unsqueeze(0)  # (1, P)

        x = self.token_emb(phoneme_ids) + self.pos_emb(positions)  # (B, P, d_model)
        x = self.dropout(self.norm(self.proj(x)))
        return x
