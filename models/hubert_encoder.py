from dataclasses import dataclass
import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

@dataclass
class HubertConfig:
    model_name: str = "facebook/hubert-base-ls960"
    device: str = "cpu"
    

class HubertEncoder:
    """
    Produces frame-level embeddings from waveform.
    Output: (T, D) float32
    """
    def __init__(self, cfg: HubertConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(cfg.model_name)
        self.model = HubertModel.from_pretrained(cfg.model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, audio: np.ndarray, sr: int = 16000) -> tuple[np.ndarray, float]:
        """
        Returns:
          embeddings: (T, D)
          frame_hz: approximate frame rate of embeddings (frames/second)
        """
        inputs = self.fe(audio, sampling_rate=sr, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)

        out = self.model(input_values)
        # last_hidden_state: (B, T, D)
        emb = out.last_hidden_state[0].detach().cpu().float().numpy()

        # Rough frame rate estimate: HuBERT uses CNN feature extractor stride
        # Often ~50Hz (20ms) or ~100Hz (10ms) depending on model config.
        # We'll estimate by ratio length / duration.
        duration = len(audio) / sr
        frame_hz = emb.shape[0] / max(duration, 1e-6)

        return emb, float(frame_hz)