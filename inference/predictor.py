from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
import torch
from transformers import AutoTokenizer
import importlib

from utils.preprocessing import preprocess_wav
from utils.prosody import basic_prosody_features, prosody_to_vector

from models.constants import SENT_DIMS, SENT_SCALE
from models.hubert_multitask import HubertMultiTask


@dataclass
class PredictorConfig:
    device:              str            = "cpu"
    hubert_name:         str            = "facebook/hubert-base-ls960"
    whisper_size:        str            = "small"
    whisper_device:      str            = "cpu"
    whisper_compute_type: str           = "int8"
    checkpoint_path:     Optional[str]  = None
    text_model_name:     Optional[str]  = None
    """
    Path to a trained HubertMultiTask checkpoint (.pt file).
    If None, the model runs with random weights — scores are not meaningful.
    """


class PronunciationPredictor:
    """
    End-to-end pronunciation scoring using a fine-tuned HubertMultiTask model.

    Pipeline:
        1. Preprocess wav (resample, VAD, normalize)
        2. Feature-extract for HuBERT
        3. ASR for word timestamps (Faster-Whisper)
        4. HubertMultiTask.forward_inference() with ASR-aligned word spans
        5. Scale outputs from [0, 1] → [0, 10] (SpeechOcean scale)
        6. Return structured JSON matching dataset annotation schema
    """

    def __init__(self, cfg: PredictorConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)

        # Text tokenizer for linguistic embedding stream
        self.text_tokenizer = (
            AutoTokenizer.from_pretrained(cfg.text_model_name)
            if cfg.text_model_name else None
        )

        # Unified model
        self.model = HubertMultiTask(
            model_name=cfg.hubert_name,
            dropout=0.0,        # dropout off at inference
            freeze_fe=True,
            text_model_name=cfg.text_model_name or None,
            freeze_text_encoder=True,
        ).to(self.device)

        if cfg.checkpoint_path is not None:
            state = torch.load(cfg.checkpoint_path, map_location=self.device, weights_only=True)
            model_state = self.model.state_dict()
            compatible = {
                k: v for k, v in state.items()
                if k in model_state and model_state[k].shape == v.shape
            }
            self.model.load_state_dict(compatible, strict=False)
            skipped = len(state) - len(compatible)
            if skipped > 0:
                self._scoring_validity = f"trained_partial:{cfg.checkpoint_path} (skipped={skipped})"
            else:
                self._scoring_validity = f"trained:{cfg.checkpoint_path}"
        else:
            self._scoring_validity = "untrained_random_init"

        self.model.eval()

        # ASR aligner for word timestamps.
        # Reloading the module here avoids stale class bindings in long-lived notebook kernels.
        asr_mod = importlib.import_module("models.asr_aligner")
        asr_mod = importlib.reload(asr_mod)
        self.aligner = asr_mod.ASRAligner(
            asr_mod.ASRConfig(
                model_size=cfg.whisper_size,
                device=cfg.whisper_device,
                compute_type=cfg.whisper_compute_type,
            )
        )

    def predict(self, wav_path: str, language: str = "en") -> Dict[str, Any]:
        # ---- 1. Preprocess ----
        audio, sr = preprocess_wav(wav_path, target_sr=16000, use_vad=True)
        audio_duration = len(audio) / sr

        # ---- 2. Normalize and convert to tensor (replaces Wav2Vec2FeatureExtractor) ----
        audio_norm   = (audio - audio.mean()) / (audio.std() + 1e-7)
        input_values = torch.tensor(audio_norm, dtype=torch.float32) \
                           .unsqueeze(0).to(self.device)       # (1, T_samples)

        # ---- 3. ASR word timestamps ----
        asr = self.aligner.transcribe_with_timestamps(
            audio, sr=sr, language=language
        )

        # ---- 4. Prosody features — computed once, used in both model and metadata ----
        p_feats = basic_prosody_features(audio, sr=sr)
        p_vec   = prosody_to_vector(p_feats)                          # normalized (5,)
        prosody_tensor = torch.tensor(p_vec, dtype=torch.float32) \
                              .unsqueeze(0).to(self.device)           # (1, 5)

        # ---- 5. Tokenize ASR transcript for text embedding stream ----
        text_input_ids      = None
        text_attention_mask = None
        if self.text_tokenizer is not None:
            text_enc = self.text_tokenizer(
                asr["text"],
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            text_input_ids      = text_enc["input_ids"].to(self.device)
            text_attention_mask = text_enc["attention_mask"].to(self.device)

        # ---- 6. Model inference (ASR-aligned spans + prosody + text) ----
        out = self.model.forward_inference(
            input_values=input_values,
            word_timestamps=asr["words"],
            audio_duration=audio_duration,
            prosody_feats=prosody_tensor,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
        )
        # out keys: sent_pred (5,), word_pred list[float], word_spans list[dict], frame_hz

        # ---- 7. Scale sentence scores [0,1] → [0,10] ----
        sent_pred = out["sent_pred"]  # np.ndarray (5,), order = SENT_DIMS
        sent_scaled = {dim: float(sent_pred[i]) * SENT_SCALE for i, dim in enumerate(SENT_DIMS)}

        # ---- 8. Build word output list ----
        # Word-level scoring is not available in the utterance-level model.
        # Word spans are retained for timestamp metadata only.
        word_spans = out["word_spans"]

        words_out: List[Dict[str, Any]] = []
        for span in word_spans:
            words_out.append({
                "text":              span["word"],
                "accuracy":          None,   # word-level model not yet trained
                "stress":            None,   # requires phone-level forced alignment
                "total":             None,   # word-level model not yet trained
                "phones":            [],     # requires forced aligner
                "phones-accuracy":   [],     # requires forced aligner
                "mispronunciations": [],
                "start":             span["start_s"],
                "end":               span["end_s"],
                "asr_prob":          span["asr_prob"],
            })

        # ---- 9. Return structured output ----
        return {
            "accuracy":    sent_scaled["accuracy"],
            "completeness": sent_scaled["completeness"],
            "fluency":     sent_scaled["fluency"],
            "prosodic":    sent_scaled["prosodic"],
            "total":       sent_scaled["total"],
            "text":        asr["text"],
            "words":       words_out,
            "audio": {
                "path": wav_path,
            },
            "inference_metadata": {
                "language":        asr.get("language"),
                "duration":        asr.get("duration"),
                "frame_hz":        out["frame_hz"],
                "prosody_features": p_feats,
                "scoring_validity": self._scoring_validity,
            },
        }
