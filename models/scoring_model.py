from __future__ import annotations

"""
HubertScoringModel: end-to-end pronunciation scoring via HuBERT + cross-attention.

Architecture::

    Audio waveform (16 kHz)
        → HuBERT encoder (layer-weighted hidden state sum)
        → Linear projection → d_model
        → Optional audio Transformer (num_audio_transformer_layers layers)
        → CrossAttentionFusion (Q=audio, K/V=phoneme embeddings)
        → Transformer encoder (num_fusion_transformer_layers layers)
        → mean pool over time
        → MLPScoringHead → 5 utterance scores in [0, 1]
        └→ Prosody auxiliary head → 5 prosodic features
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import HubertModel

from models.phoneme_embedder import PhonemeEmbedder
from models.phoneme_vocab import VOCAB_SIZE
from models.scoring_heads import CrossAttentionFusion, MLPScoringHead


class HubertScoringModel(nn.Module):
    """Multi-task pronunciation scoring model.

    Encodes audio with a layer-weighted HuBERT backbone, attends over
    reference phoneme embeddings via cross-attention, and predicts five
    sentence-level pronunciation scores plus five prosodic features.

    Args:
        model_name: HuggingFace model ID for HuBERT.
        d_model: Internal embedding dimension used throughout the model.
        num_heads: Number of attention heads in all Transformer components.
        num_audio_transformer_layers: Layers of audio-only Transformer encoder
            applied after the linear projection (0 = skip entirely).
        num_fusion_transformer_layers: Layers of post-fusion Transformer encoder.
        mlp_hidden_layers: Hidden ``Linear → ReLU → Dropout`` blocks in the
            MLP scoring head.
        dropout: Dropout probability used throughout.
        freeze_feature_extractor: Whether to freeze the HuBERT CNN feature
            extractor weights.
        num_unfreeze_hubert_layers: Number of trailing HuBERT Transformer
            encoder layers to keep trainable; all others are frozen.
        num_aspects: Number of sentence-level pronunciation score aspects.
        prosody_feat_dim: Dimensionality of the prosody auxiliary target.
    """

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        d_model: int = 256,
        num_heads: int = 8,
        num_audio_transformer_layers: int = 1,
        num_fusion_transformer_layers: int = 2,
        mlp_hidden_layers: int = 2,
        dropout: float = 0.1,
        freeze_feature_extractor: bool = True,
        num_unfreeze_hubert_layers: int = 12,
        num_aspects: int = 5,
        prosody_feat_dim: int = 5,
    ) -> None:
        super().__init__()

        # ── HuBERT backbone ──────────────────────────────────────────────────
        self.hubert: HubertModel = HubertModel.from_pretrained(model_name)

        if freeze_feature_extractor:
            self.hubert.feature_extractor._freeze_parameters()

        encoder_layers = self.hubert.encoder.layers
        for layer in encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
        if num_unfreeze_hubert_layers > 0:
            for layer in encoder_layers[-num_unfreeze_hubert_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # ── Layer-weighted pooling ───────────────────────────────────────────
        num_hidden_layers: int = self.hubert.config.num_hidden_layers
        self.layer_weights = nn.Parameter(torch.ones(num_hidden_layers + 1))

        # ── Audio projection ─────────────────────────────────────────────────
        hubert_hidden_size: int = self.hubert.config.hidden_size
        self.audio_proj = nn.Linear(hubert_hidden_size, d_model)

        # ── Phoneme embedder ─────────────────────────────────────────────────
        self.phoneme_embedder = PhonemeEmbedder(d_model, VOCAB_SIZE, dropout=dropout)

        # ── Optional audio Transformer ───────────────────────────────────────
        if num_audio_transformer_layers > 0:
            audio_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.audio_transformer: Optional[nn.TransformerEncoder] = nn.TransformerEncoder(
                audio_layer,
                num_layers=num_audio_transformer_layers,
                enable_nested_tensor=False,
            )
        else:
            self.audio_transformer = None

        # ── Cross-attention fusion ───────────────────────────────────────────
        self.cross_attn = CrossAttentionFusion(d_model, num_heads, dropout)

        # ── Post-fusion Transformer ──────────────────────────────────────────
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            fusion_layer,
            num_layers=num_fusion_transformer_layers,
            enable_nested_tensor=False,
        )

        # ── Output heads ─────────────────────────────────────────────────────
        self.scoring_head = MLPScoringHead(d_model, num_aspects, mlp_hidden_layers, dropout)

        self.prosody_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, prosody_feat_dim),
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _encode_audio(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run HuBERT and return a weighted sum of all hidden states.

        Args:
            input_values: ``(B, num_samples)`` raw 16 kHz waveform.
            attention_mask: ``(B, num_samples)`` optional padding mask.

        Returns:
            ``(B, T, hubert_hidden_size)`` layer-weighted hidden states.
        """
        out = self.hubert(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states: tuple of (num_hidden_layers+1) tensors each (B, T, H)
        stacked = torch.stack(out.hidden_states, dim=0)           # (n+1, B, T, H)
        weights = torch.softmax(self.layer_weights, dim=0)         # (n+1,)
        return (weights[:, None, None, None] * stacked).sum(dim=0)  # (B, T, H)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_values: torch.Tensor,
        phoneme_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        phoneme_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the full pronunciation scoring pipeline.

        Args:
            input_values: ``(B, num_samples)`` raw 16 kHz waveform tensor.
            phoneme_ids: ``(B, P)`` phoneme vocabulary ID tensor.
            attention_mask: ``(B, num_samples)`` mask where 1 = valid, 0 = pad.
            phoneme_mask: ``(B, P)`` bool mask where ``True`` = valid token.
                Inverted before passing to cross-attention (``True`` = ignore
                in :class:`~torch.nn.MultiheadAttention`).

        Returns:
            Dict containing:

            - ``"sent_pred"``: ``(B, num_aspects)`` scores in ``[0, 1]``.
            - ``"prosody_pred"``: ``(B, prosody_feat_dim)`` raw prosody outputs.
        """
        audio = self._encode_audio(input_values, attention_mask)  # (B, T, H)
        audio = self.audio_proj(audio)                             # (B, T, d_model)

        if self.audio_transformer is not None:
            audio = self.audio_transformer(audio)

        phoneme_emb = self.phoneme_embedder(phoneme_ids)  # (B, P, d_model)

        # MultiheadAttention treats True as "ignore" — invert valid→ignore mask
        key_padding_mask: Optional[torch.Tensor] = None
        if phoneme_mask is not None:
            key_padding_mask = ~phoneme_mask.bool()

        fused = self.cross_attn(audio, phoneme_emb, key_padding_mask)  # (B, T, d_model)
        fused = self.fusion_transformer(fused)                          # (B, T, d_model)

        pooled = fused.mean(dim=1)  # (B, d_model)

        return {
            "sent_pred": self.scoring_head(pooled),
            "prosody_pred": self.prosody_head(pooled),
        }
