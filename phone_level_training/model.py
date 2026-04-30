import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        pe = torch.zeros(500, d_model)
        pos = torch.arange(0, 500).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class PhoneModel(nn.Module):
    def __init__(self, num_phones, input_dim_without_embed=1026):
        super().__init__()
        self.embedding = nn.Embedding(num_phones, 64)

        self.proj = nn.Linear(input_dim_without_embed + 64, 256)
        self.pos = PositionalEncoding(256)

        layer = nn.TransformerEncoderLayer(256, 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 4)

        self.head = nn.Linear(256, 1)

    def forward(self, x, mask):
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=~mask)
        return self.head(x).squeeze(-1)