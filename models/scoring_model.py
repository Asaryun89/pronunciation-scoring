from __future__ import annotations

"""
HubertScoringModel: end-to-end pronunciation scoring via HuBERT + BGE cross-attention.

Architecture::

    Audio waveform (16 kHz)
        → HuBERT encoder (layer-weighted hidden state sum)
        → Linear projection → d_model
        → Optional audio Transformer (num_audio_transformer_layers layers)
        → CrossAttentionFusion (Q=audio, K/V=text sentence embedding)
        → Transformer encoder (num_fusion_transformer_layers layers)
        → mean pool over time
        → MLPScoringHead → 5 utterance scores in [0, 1]
        └→ Prosody auxiliary head → 5 prosodic features

# TODO(phase2): Swap HuBERT backbone to HuBERT-CTC from Linh's checkpoint
# See: branch linh/hubert_only, model: hubert-large-speechocean-ctc
# Requires: update audio_proj dim 768 → 1024, layer_weights size 13 → 25

"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from models.scoring_heads import CrossAttentionFusion, MLPScoringHead
from models.text_embedder import BGETextEmbedder


class HubertScoringModel(nn.Module):
    """Multi-task pronunciation scoring model with BGE text branch.

    Encodes audio with a layer-weighted HuBERT backbone, conditions on a
    BGE sentence embedding via cross-attention, and predicts five
    sentence-level pronunciation scores plus five prosodic features.

    Args:
        hubert_model_name: HuggingFace model ID for HuBERT.
        bge_model_name: HuggingFace model ID for BGE text encoder.
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
        freeze_bge: Whether to freeze all BGE encoder parameters.
        num_aspects: Number of sentence-level pronunciation score aspects.
        prosody_feat_dim: Dimensionality of the prosody auxiliary target.
    """

    def __init__(
        self,
        hubert_model_name: str = "facebook/hubert-base-ls960",
        bge_model_name: str = "BAAI/bge-small-en-v1.5",
        d_model: int = 256,
        num_heads: int = 8,
        num_audio_transformer_layers: int = 1,
        num_fusion_transformer_layers: int = 2,
        mlp_hidden_layers: int = 2,
        dropout: float = 0.1,
        freeze_feature_extractor: bool = True,
        num_unfreeze_hubert_layers: int = 12,
        freeze_bge: bool = True,
        num_aspects: int = 5,
        prosody_feat_dim: int = 5,
    ) -> None:
        super().__init__()

        # ── HuBERT backbone ──────────────────────────────────────────────────
        self.hubert = AutoModel.from_pretrained(hubert_model_name)

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

        # ── BGE text embedder ────────────────────────────────────────────────
        self.text_embedder = BGETextEmbedder(d_model, bge_model_name, freeze_encoder=freeze_bge, dropout=dropout)

        # ── Optional audio Transformer ───────────────────────────────────────
        if num_audio_transformer_layers > 0:
            audio_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                enable_nested_tensor=False,
            )
            self.audio_transformer: Optional[nn.TransformerEncoder] = nn.TransformerEncoder(
                audio_layer,
                num_layers=num_audio_transformer_layers,
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
            enable_nested_tensor=False,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            fusion_layer,
            num_layers=num_fusion_transformer_layers,
        )

        # ── Output heads ─────────────────────────────────────────────────────
        self.scoring_head = MLPScoringHead(d_model, num_aspects, mlp_hidden_layers, dropout)

        self.prosody_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, prosody_feat_dim),
        )

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
        stacked = torch.stack(out.hidden_states, dim=0)            # (n+1, B, T, H)
        weights = torch.softmax(self.layer_weights, dim=0)          # (n+1,)
        return (weights[:, None, None, None] * stacked).sum(dim=0)  # (B, T, H)

    def forward(
        self,
        input_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        **_,
    ) -> Dict[str, torch.Tensor]:
        """Run the full pronunciation scoring pipeline.

        Args:
            input_values: ``(B, num_samples)`` raw 16 kHz waveform tensor.
            text_input_ids: ``(B, L)`` tokenised script.
            text_attention_mask: ``(B, L)`` 1=valid, 0=pad.
            audio_attention_mask: ``(B, num_samples)`` optional audio padding mask.

        Returns:
            Dict containing:

            - ``"sent_pred"``: ``(B, 5)`` scores in ``[0, 1]``.
            - ``"prosody_pred"``: ``(B, 5)`` raw prosody outputs.
        """
        audio = self._encode_audio(input_values, audio_attention_mask)  # (B, T, H)
        audio = self.audio_proj(audio)                                   # (B, T, d_model)

        if self.audio_transformer is not None:
            audio = self.audio_transformer(audio)

        # text_emb: (B, 1, d_model) — L=1, no padding mask needed
        text_emb = self.text_embedder(text_input_ids, text_attention_mask)

        fused = self.cross_attn(audio, text_emb)          # (B, T, d_model)
        fused = self.fusion_transformer(fused)             # (B, T, d_model)

        pooled = fused.mean(dim=1)  # (B, d_model)

        return {
            "sent_pred": self.scoring_head(pooled),
            "prosody_pred": self.prosody_head(pooled),
        }
