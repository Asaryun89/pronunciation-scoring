from utils.preprocessing import preprocess_wav
from models.hubert_encoder import HubertEncoder, HubertConfig

# @dataclass
# class PredictorConfig:
#     device: str = "cpu"
#     hubert_name: str = "facebook/hubert-base-ls960"
#     whisper_size: str = "small"
#     whisper_device: str = "cpu"
#     whisper_compute_type: str = "int8"

wav_path = r"C:\Users\Admin\Documents\pronunciation_scoring\pronunciation-scoring\audio\native_speaker\02_native.wav"
audio, sr = preprocess_wav(wav_path, target_sr=16000, use_vad=True)
encoder = HubertEncoder(HubertConfig(model_name="facebook/hubert-base-ls960", device="cpu"))
emb, frame_hz = encoder.encode(audio, sr=sr)  # (T,D)
