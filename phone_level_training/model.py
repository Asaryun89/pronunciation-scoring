import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        pe = torch.zeros(500, d_model)
        pos = torch.arange(0, 500).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PhoneModel(nn.Module):
    """Transformer scorer for per-phoneme accuracy (target scale 0–2).

    All HuBERT transformer layers are used via a learnable softmax-weighted
    sum, initialised uniformly (zeros before softmax → equal weights at t=0).
    Ablation flags let you zero out feature groups to measure their contribution.

    Args:
        num_phones:        vocabulary size for the phoneme-identity embedding.
        ssl_dim:           hidden size of the HuBERT model (768 for base, 1024 for large).
        num_hubert_layers: number of HuBERT transformer layers (12 for base, 24 for large).
        use_ssl:           include HuBERT layer-weighted SSL embeddings.
        use_gop:           include 1-dim GOP score.
        use_dur:           include 1-dim log-duration.
        use_phone_embed:   include 64-dim learnable phoneme-identity embedding.
    """

    def __init__(
        self,
        num_phones: int,
        ssl_dim: int = 768,
        num_hubert_layers: int = 12,
        use_ssl: bool = True,
        use_gop: bool = True,
        use_dur: bool = True,
        use_phone_embed: bool = True,
    ):
        super().__init__()
        self.use_ssl = use_ssl
        self.use_gop = use_gop
        self.use_dur = use_dur
        self.use_phone_embed = use_phone_embed

        self.ssl_dim = ssl_dim
        input_dim = 0
        if use_ssl:
            # Learnable per-layer weights; zeros → uniform softmax at init.
            self.layer_weights = nn.Parameter(torch.zeros(num_hubert_layers))
            input_dim += ssl_dim
        if use_gop:
            input_dim += 1
        if use_dur:
            input_dim += 1
        if use_phone_embed:
            self.embedding = nn.Embedding(num_phones, 64)
            input_dim += 64

        assert input_dim > 0, "At least one feature group must be enabled."

        self.proj = nn.Linear(input_dim, 256)
        self.pos = PositionalEncoding(256)

        layer = nn.TransformerEncoderLayer(256, 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 4)

        self.head = nn.Linear(256, 1)

    def forward(self, ssl, gop, dur, phone_ids, mask):
        """
        Args:
            ssl:       (B, T, num_layers, 1024)
            gop:       (B, T, 1)
            dur:       (B, T, 1)
            phone_ids: (B, T)   long
            mask:      (B, T)   bool, True = valid token

        Returns:
            (B, T) predicted per-phoneme accuracy scores in [0, 1]
            (multiply by PHONE_SCALE=2 for display)
        """
        parts = []
        if self.use_ssl:
            # Weighted sum across layers: (B, T, num_layers, 1024) → (B, T, 1024)
            weights = torch.softmax(self.layer_weights, dim=0)  # (num_layers,)
            ssl_weighted = (ssl * weights.view(1, 1, -1, 1)).sum(dim=2)
            parts.append(ssl_weighted)
        if self.use_gop:
            parts.append(gop)
        if self.use_dur:
            parts.append(dur)
        if self.use_phone_embed:
            parts.append(self.embedding(phone_ids))

        x = torch.cat(parts, dim=-1)          # (B, T, input_dim)
        x = self.proj(x)                       # (B, T, 256)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=~mask)
        return self.head(x).squeeze(-1)        # (B, T)
