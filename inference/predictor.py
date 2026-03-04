from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import torch

from utils.preprocessing import preprocess_wav
from utils.alignment import build_word_segments
from utils.pooling import mean_pool, utt_pool_mean_std
from utils.prosody import basic_prosody_features, prosody_to_vector
from utils.fusion import fuse_scores

from models.hubert_encoder import HubertEncoder, HubertConfig
from models.asr_aligner import ASRAligner, ASRConfig
from models.scoring_heads import MultiHeadScorer, ScoringConfig

@dataclass
class PredictorConfig:
    device: str = "cpu"
    hubert_name: str = "facebook/hubert-base-ls960"
    whisper_size: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

class PronunciationPredictor:
    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg

        self.encoder = HubertEncoder(HubertConfig(model_name=cfg.hubert_name, device=cfg.device))
        self.aligner = ASRAligner(ASRConfig(model_size=cfg.whisper_size, device=cfg.whisper_device, compute_type=cfg.whisper_compute_type))

        # Build scorer with correct dims after we know encoder dim
        # We'll init lazily on first call.
        self.scorer = None
        self.scorer_device = torch.device(cfg.device)

    def _lazy_init_scorer(self, emb_dim: int):
        utt_dim = emb_dim * 2  # mean+std pooling
        scfg = ScoringConfig(emb_dim=emb_dim, utt_dim=utt_dim, prosody_dim=5, hidden=256)
        self.scorer = MultiHeadScorer(scfg).to(self.scorer_device)
        self.scorer.eval()
        # NOTE: This is an untrained scorer by default.
        # Replace weights by loading your trained checkpoint in real usage.

    @torch.inference_mode()
    def predict(self, wav_path: str, language: str = "en") -> Dict[str, Any]:
        audio, sr = preprocess_wav(wav_path, target_sr=16000, use_vad=True)

        emb, frame_hz = self.encoder.encode(audio, sr=sr)  # (T,D)
        T, D = emb.shape

        if self.scorer is None:
            self._lazy_init_scorer(D)

        asr = self.aligner.transcribe_with_timestamps(audio, sr=sr, language=language)
        word_segments = build_word_segments(asr["words"], frame_hz=frame_hz, T=T)

        # Word embeddings
        z_words = []
        words_out = []
        for seg in word_segments:
            z = mean_pool(emb, seg["i0"], seg["i1"])
            z_words.append(z)
            words_out.append({
                "word": seg["word"],
                "start": seg["start_s"],
                "end": seg["end_s"],
                "asr_prob": seg["asr_prob"]
            })

        # Utterance embedding
        z_utt = utt_pool_mean_std(emb)

        # Prosody vector
        p_feats = basic_prosody_features(audio, sr=sr)
        p_vec = prosody_to_vector(p_feats)

        # Torch tensors
        if len(z_words) > 0:
            z_words_t = torch.tensor(np.stack(z_words), device=self.scorer_device)
            word_scores = self.scorer.forward_word(z_words_t).detach().cpu().numpy().tolist()
            mean_word = float(np.mean(word_scores))
        else:
            word_scores = []
            mean_word = 0.0

        z_utt_t = torch.tensor(z_utt[None, :], device=self.scorer_device)
        utt_score = float(self.scorer.forward_utt(z_utt_t).detach().cpu().item())

        p_t = torch.tensor(p_vec[None, :], device=self.scorer_device)
        prosody_score = float(self.scorer.forward_prosody(p_t).detach().cpu().item())

        final = fuse_scores(utt_score_0_100=utt_score, mean_word_0_1=mean_word, prosody_0_1=prosody_score)

        # Attach word scores
        for i in range(len(words_out)):
            words_out[i]["score"] = float(word_scores[i]) if i < len(word_scores) else None

        return {
            "text": asr["text"],
            "overall_score": final,
            "accuracy": mean_word * 100.0,
            "fluency": utt_score,
            "prosody": prosody_score * 100.0,
            "prosody_features": p_feats,
            "words": words_out
        }