"""
HubertMultiTask — pronunciation scoring model.

Architecture (matches the diagram):

  Audio path
  ----------
  Raw audio → HuBERT encoder (frozen CNN / fine-tuned transformer)
            → Layer-weighted sum over all hidden states   (learnable weights)
            → Linear projection                           → (B, T, d_model)

  Text path  (optional)
  ---------
  Reference text → BERT encoder (frozen or fine-tuned)
                 → CLS token / mean pooling              → (B, text_H)
                 → Linear projection + LayerNorm          → (B, 1, d_model)

  Phoneme aligner  (optional dashed path)
  ----------------
  Raw audio + transcript → CTC forced alignment → phone spans
  Phone-level pooled audio embeddings appended to text K/V sequence

  Fusion & refinement
  -------------------
  Cross-attention fusion  — Audio Q, Text(+phone) K/V
  Conformer / Transformer block  — post-fusion contextual refinement

  Head
  ----
  Mean-pool over time → MLP: FC → ReLU → Dropout → FC → Sigmoid × 5

  Auxiliary
  ---------
  Prosody feature head  — FC → ReLU → FC  (training regulariser, w_pfeat weight)
"""

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import HubertModel, AutoModel

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from stage2_scoring.scoring_heads import CrossAttentionFusion, MLPScoringHead  # noqa: E402
from utils.alignment import build_word_segments                         # noqa: E402


class HubertMultiTask(nn.Module):
    """
    Unified pronunciation scoring model for training and inference.

    Parameters
    ----------
    model_name                  : HuBERT checkpoint (HuggingFace hub or local)
    d_model                     : common projection dimension shared by both streams
    num_heads                   : attention heads in all attention blocks
    num_audio_transformer_layers: pre-fusion audio self-attention depth (0 = disabled)
    num_transformer_layers      : post-fusion Transformer encoder depth
    mlp_hidden_layers           : hidden FC→ReLU→Dropout blocks in the scoring head
    dropout                     : dropout rate used throughout
    freeze_fe                   : freeze HuBERT CNN feature extractor
    num_unfreeze_hubert_layers  : unfreeze the top-N HuBERT transformer layers (0 = all frozen)
    text_model_name             : embedding model for the text stream ('' / None = disabled)
    freeze_text_encoder         : keep text encoder weights frozen during training
    prosody_feat_dim            : auxiliary prosody output dimension (default 5)
    """

    def __init__(
        self,
        model_name:                   str           = "facebook/hubert-large-ll60k",
        d_model:                      int           = 256,
        num_heads:                    int           = 8,
        num_audio_transformer_layers: int           = 1,
        num_transformer_layers:       int           = 2,
        mlp_hidden_layers:            int           = 2,
        dropout:                      float         = 0.1,
        freeze_fe:                    bool          = False,
        num_unfreeze_hubert_layers:   int           = 12,
        text_model_name:              Optional[str] = "Qwen/Qwen3-Embedding-0.6B",
        freeze_text_encoder:          bool          = True,
        prosody_feat_dim:             int           = 5,
    ):
        super().__init__()

        # ── HuBERT backbone ───────────────────────────────────────────────
        self.hubert = HubertModel.from_pretrained(model_name)
        if freeze_fe:
            self.hubert.feature_extractor._freeze_parameters()

        # Partial HuBERT unfreezing: top-N transformer encoder layers become trainable.
        # All other transformer parameters remain frozen (requires_grad stays False from
        # HuggingFace default — HuBERT is fully trainable by default, so we flip it).
        if num_unfreeze_hubert_layers < self.hubert.config.num_hidden_layers:
            for p in self.hubert.encoder.parameters():
                p.requires_grad = False
            if num_unfreeze_hubert_layers > 0:
                for layer in self.hubert.encoder.layers[-num_unfreeze_hubert_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

        H        = self.hubert.config.hidden_size       # 1024 for hubert-large
        n_layers = self.hubert.config.num_hidden_layers + 1  # transformer layers + embedding layer

        # Learnable scalar weights for the layer-weighted sum.
        # Softmax is applied at runtime so they always form a valid convex combination.
        self.layer_weights = nn.Parameter(torch.ones(n_layers))

        # Audio linear projection: H → d_model
        self.audio_proj = nn.Linear(H, d_model)

        # ── Optional text encoder (BERT-style) ────────────────────────────
        self.text_encoder = None
        self.text_proj    = None
        if text_model_name:
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            text_H = self.text_encoder.config.hidden_size
            if freeze_text_encoder:
                for p in self.text_encoder.parameters():
                    p.requires_grad = False
            # Text linear projection: text_H → d_model
            self.text_proj = nn.Sequential(
                nn.Linear(text_H, d_model),
                nn.LayerNorm(d_model),
            )

        # ── Pre-fusion audio self-attention block ─────────────────────────
        # Refines audio representations in the d_model space before cross-attention.
        # Disabled (identity pass-through) when num_audio_transformer_layers == 0.
        if num_audio_transformer_layers > 0:
            _audio_enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads,
                dim_feedforward=4 * d_model, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            self.audio_transformer: Optional[nn.TransformerEncoder] = nn.TransformerEncoder(
                _audio_enc, num_layers=num_audio_transformer_layers,
                enable_nested_tensor=False,
            )
        else:
            self.audio_transformer = None

        # ── Cross-attention fusion (Audio Q, Text K/V) ────────────────────
        self.cross_attn_fusion = CrossAttentionFusion(d_model, num_heads, dropout)

        # ── Post-fusion Transformer block (Conformer equivalent) ──────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN: more stable gradient flow
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers,
            enable_nested_tensor=False,   # pre-LN (norm_first=True) disables nested tensors anyway
        )

        # ── MLP scoring head: (FC → ReLU → Dropout) × mlp_hidden_layers → FC → Sigmoid × 5
        self.scorer = MLPScoringHead(
            d_model, num_aspects=5, hidden_layers=mlp_hidden_layers, dropout=dropout
        )

        # ── Auxiliary prosody feature head (training regulariser) ─────────
        self.prosody_feat_dim  = prosody_feat_dim
        self.prosody_feat_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, prosody_feat_dim),
        )

        # Stored for forward_inference
        self._d_model = d_model

    # ------------------------------------------------------------------
    # Internal encoders
    # ------------------------------------------------------------------

    def _layer_weighted_encode(
        self,
        input_values:   torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run HuBERT with all hidden states and return a learnable weighted sum.

        Returns (B, T_frames, H).
        """
        out = self.hubert(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states: tuple of tensors, each (B, T, H)
        # length = num_hidden_layers + 1  (embedding layer + transformer layers)
        stacked = torch.stack(out.hidden_states, dim=0)   # (n_layers, B, T, H)
        weights = torch.softmax(self.layer_weights, dim=0) # (n_layers,)  sum to 1
        return (weights[:, None, None, None] * stacked).sum(dim=0)  # (B, T, H)

    def _encode_text(
        self,
        text_input_ids:      Optional[torch.Tensor],  # (B, L)
        text_attention_mask: Optional[torch.Tensor],  # (B, L)
    ) -> Optional[torch.Tensor]:
        """
        Run the text encoder and return a sentence embedding via mask-aware mean pooling.

        Mean pooling over non-padding tokens is correct for decoder-style embedding
        models (e.g. Qwen3-Embedding) that do not have a special CLS token.

        Returns (B, text_H) or None when the text stream is disabled.
        """
        if self.text_encoder is None or text_input_ids is None:
            return None
        out = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        hidden = out.last_hidden_state  # (B, L, text_H)
        if text_attention_mask is not None:
            mask   = text_attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return hidden.mean(dim=1)  # (B, text_H)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_values:        torch.Tensor,
        attention_mask:      Optional[torch.Tensor] = None,
        text_input_ids:      Optional[torch.Tensor] = None,   # (B, L)
        text_attention_mask: Optional[torch.Tensor] = None,   # (B, L)
        # Legacy collator keys (word_mask, phone_mask, word_frame_spans, prosody_feats)
        # are absorbed here so existing train.py call-sites need no changes.
        **_,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass used during training.

        Returns
        -------
        sent_pred    (B, 5)              — utterance scores in [0, 1]
        prosody_pred (B, prosody_feat_dim) — auxiliary prosody prediction (unbounded)
        """
        # ── Audio path ────────────────────────────────────────────────
        hidden    = self._layer_weighted_encode(input_values, attention_mask)  # (B, T, H)
        audio_emb = self.audio_proj(hidden)                                    # (B, T, d_model)

        # ── Text path ─────────────────────────────────────────────────
        # Mean pool → Linear+LN → (B, 1, d_model) used as K/V in cross-attention
        text_emb_kv = None
        if self.text_encoder is not None and text_input_ids is not None:
            text_emb    = self._encode_text(
                text_input_ids.to(hidden.device),
                text_attention_mask.to(hidden.device) if text_attention_mask is not None else None,
            )                                               # (B, text_H)
            text_emb_kv = self.text_proj(text_emb).unsqueeze(1)  # (B, 1, d_model)

        # ── Pre-fusion audio self-attention ───────────────────────────
        if self.audio_transformer is not None:
            audio_emb = self.audio_transformer(audio_emb)

        # ── Cross-attention fusion (Audio Q, Text K/V) ────────────────
        if text_emb_kv is not None:
            fused = self.cross_attn_fusion(audio_emb, text_emb_kv)
        else:
            fused = audio_emb

        # ── Post-fusion Transformer block ─────────────────────────────
        refined = self.transformer(fused)         # (B, T, d_model)

        # ── Utterance-level mean pooling ──────────────────────────────
        utt_emb = refined.mean(dim=1)             # (B, d_model)

        # ── Scoring head ──────────────────────────────────────────────
        sent_pred    = self.scorer(utt_emb)                       # (B, 5)
        prosody_pred = self.prosody_feat_head(utt_emb)            # (B, prosody_feat_dim)

        return {
            "sent_pred":    sent_pred,
            "prosody_pred": prosody_pred,
        }

    # ------------------------------------------------------------------
    # Inference forward
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward_inference(
        self,
        input_values:        torch.Tensor,
        word_timestamps:     List[Dict[str, Any]],
        audio_duration:      float,
        text_input_ids:      Optional[torch.Tensor] = None,   # (1, L)
        text_attention_mask: Optional[torch.Tensor] = None,   # (1, L)
        phone_spans:         Optional[List]         = None,   # from CTCPhoneAligner (optional)
        **_,   # absorbs prosody_feats, max_phones_per_word from legacy call-sites
    ) -> Dict[str, Any]:
        """
        Run inference with ASR word-boundary timestamps.

        Parameters
        ----------
        input_values     : (1, T_samples) — HuBERT feature-extractor output
        word_timestamps  : list of dicts with 'word', 'start', 'end', 'probability'
        audio_duration   : duration of the utterance in seconds
        text_input_ids   : tokenized transcript for the text stream
        text_attention_mask: padding mask for text tokens
        phone_spans      : optional list of PhoneSpan from CTCPhoneAligner

        Returns
        -------
        sent_pred  : np.ndarray (5,) in [0, 1], order = SENT_DIMS
        word_spans : list[dict] with i0, i1, word, start_s, end_s, asr_prob
        frame_hz   : estimated HuBERT frame rate (frames/sec)
        """
        # ── Audio path ────────────────────────────────────────────────
        hidden    = self._layer_weighted_encode(input_values)  # (1, T, H)
        audio_emb = self.audio_proj(hidden)                    # (1, T, d_model)
        T         = audio_emb.size(1)
        device    = audio_emb.device

        frame_hz   = T / max(audio_duration, 1e-6)
        word_spans = build_word_segments(word_timestamps, frame_hz=frame_hz, T=T)

        if len(word_spans) == 0:
            return {
                "sent_pred":  np.zeros(5, dtype=np.float32),
                "word_spans": [],
                "frame_hz":   frame_hz,
            }

        # ── Text path ─────────────────────────────────────────────────
        text_emb_kv = None
        if self.text_encoder is not None and text_input_ids is not None:
            text_emb    = self._encode_text(
                text_input_ids.to(device),
                text_attention_mask.to(device) if text_attention_mask is not None else None,
            )                                               # (1, text_H)
            text_emb_kv = self.text_proj(text_emb).unsqueeze(1)  # (1, 1, d_model)

        # ── Optional: augment K/V with phone-level audio embeddings ──
        # (dashed path in diagram — active when CTCPhoneAligner provides spans)
        if phone_spans and text_emb_kv is not None:
            phone_embs = _pool_phone_embeddings(audio_emb[0], phone_spans)
            if phone_embs is not None:
                # Concatenate phone embeddings along the sequence dimension
                text_emb_kv = torch.cat([text_emb_kv, phone_embs.unsqueeze(0)], dim=1)

        # ── Cross-attention fusion ────────────────────────────────────
        if text_emb_kv is not None:
            fused = self.cross_attn_fusion(audio_emb, text_emb_kv)
        else:
            fused = audio_emb

        # ── Post-fusion Transformer block ─────────────────────────────
        refined = self.transformer(fused)     # (1, T, d_model)

        # ── Utterance-level mean pooling + scoring ────────────────────
        utt_emb   = refined.mean(dim=1)       # (1, d_model)
        sent_pred = self.scorer(utt_emb)[0]   # (5,)

        return {
            "sent_pred":  sent_pred.cpu().numpy(),
            "word_spans": word_spans,
            "frame_hz":   frame_hz,
        }


# ---------------------------------------------------------------------------
# Helper — phone-level embedding pooling (optional CTC path)
# ---------------------------------------------------------------------------

def _pool_phone_embeddings(
    audio_emb:   torch.Tensor,  # (T, d_model) — single sample, already on correct device
    phone_spans: List,          # list of PhoneSpan from CTCPhoneAligner
) -> Optional[torch.Tensor]:
    """
    Pool audio embeddings within each phone span into a single vector per phone.

    Returns (n_phones, d_model) or None if no valid spans.
    """
    T = audio_emb.size(0)
    vecs = []
    for span in phone_spans:
        i0 = max(0, getattr(span, "frame_start", 0))
        i1 = min(T, getattr(span, "frame_end",   T))
        i1 = max(i1, i0 + 1)
        vecs.append(audio_emb[i0:i1].mean(dim=0))
    if not vecs:
        return None
    return torch.stack(vecs, dim=0)  # (n_phones, d_model)
