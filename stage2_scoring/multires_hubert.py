"""
MultiResHuBERT — multi-resolution audio encoder for HubertMultiTask.

Wraps a pretrained HuBERT backbone and, instead of a single learnable
layer-weighted sum over all hidden states, computes `num_streams` independent
weighted sums — each with its own softmax-normalised layer-weight vector and
projection head. Layer weights are Gaussian-initialised with peaks spread
across the network depth, so each stream starts biased toward a different
resolution (e.g. lower layers ~ acoustic/phonetic detail, higher layers ~
semantic/prosodic information) while remaining fully learnable. The streams
are concatenated and projected back to `d_model` so the rest of
HubertMultiTask (audio transformer, fusion, scorer) is unchanged.
"""

import torch
import torch.nn as nn
from transformers import HubertModel


class MultiResHuBERT(nn.Module):
    """
    Parameters
    ----------
    model_name                 : HuBERT checkpoint (HuggingFace hub or local)
    d_model                    : output projection dimension
    num_streams                : number of independent layer-weighted streams
    freeze_fe                  : freeze HuBERT CNN feature extractor
    num_unfreeze_hubert_layers : unfreeze the top-N HuBERT transformer layers
    """

    def __init__(
        self,
        model_name: str,
        d_model: int,
        num_streams: int = 4,
        freeze_fe: bool = False,
        num_unfreeze_hubert_layers: int = 12,
    ):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name)
        if freeze_fe:
            self.hubert.feature_extractor._freeze_parameters()

        if num_unfreeze_hubert_layers < self.hubert.config.num_hidden_layers:
            for p in self.hubert.encoder.parameters():
                p.requires_grad = False
            if num_unfreeze_hubert_layers > 0:
                for layer in self.hubert.encoder.layers[-num_unfreeze_hubert_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

        H        = self.hubert.config.hidden_size
        n_layers = self.hubert.config.num_hidden_layers + 1  # transformer layers + embedding layer
        self.num_streams = num_streams

        # [num_streams, n_layers] learnable layer-weight logits, Gaussian-initialised
        # with peaks spread evenly across the network depth so each stream starts
        # attending to a different range of layers (softmax applied at runtime).
        layer_idx = torch.arange(n_layers, dtype=torch.float32)
        peaks     = torch.linspace(0, n_layers - 1, num_streams)
        spread    = max(n_layers / num_streams, 1.0)
        init      = -((layer_idx[None, :] - peaks[:, None]) ** 2) / (2 * spread ** 2)
        self.layer_weights = nn.Parameter(init)  # (num_streams, n_layers)

        self.proj = nn.ModuleList([nn.Linear(H, d_model) for _ in range(num_streams)])
        self.fuse = nn.Linear(num_streams * d_model, d_model)

    def forward(
        self,
        input_values:   torch.Tensor,
        attention_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """input_values (B, T_samples) -> fused multi-resolution embedding (B, T, d_model)."""
        out = self.hubert(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        stacked = torch.stack(out.hidden_states, dim=0)       # (n_layers, B, T, H)
        weights = torch.softmax(self.layer_weights, dim=-1)   # (S, n_layers), each row sums to 1

        streams = []
        for s in range(self.num_streams):
            w     = weights[s][:, None, None, None]          # (n_layers, 1, 1, 1)
            mixed = (w * stacked).sum(dim=0)                  # (B, T, H)
            streams.append(self.proj[s](mixed))               # (B, T, d_model)

        return self.fuse(torch.cat(streams, dim=-1))          # (B, T, d_model)
