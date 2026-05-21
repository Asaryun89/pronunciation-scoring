import io
import numpy as np
import soundfile as sf
import torch
from datasets import Audio
from scipy.signal import resample as scipy_resample

from .alignment import align
from .features import extract_ssl_and_logprob, aggregate_ssl, compute_gop, compute_duration
from .config import DEVICE

TARGET_SR = 16_000


def _load_audio(audio_info: dict) -> np.ndarray:
    """Decode audio from a datasets Audio field (decode=False) using soundfile.

    Returns a float32 mono array at TARGET_SR.
    """
    raw = audio_info.get("bytes")
    if raw:
        audio, sr = sf.read(io.BytesIO(raw))
    else:
        audio, sr = sf.read(audio_info["path"])

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    if sr != TARGET_SR:
        n = int(len(audio) * TARGET_SR / sr)
        audio = scipy_resample(audio, n)

    return audio.astype(np.float32)


class PhoneDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, model, phone2id):
        # Disable automatic audio decoding so datasets doesn't call torchcodec.
        self.dataset = dataset.cast_column("audio", Audio(decode=False))
        self.model = model
        self.phone2id = phone2id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        audio = _load_audio(sample["audio"])

        phones = []
        scores = []
        for w in sample["words"]:
            phones.extend(w["phones"])
            scores.extend(w["phones-accuracy"])

        phone_ids = [self.phone2id[p] for p in phones]

        audio = (audio - audio.mean()) / (audio.std() + 1e-7)
        input_values = torch.tensor(audio).unsqueeze(0).to(DEVICE)

        ssl, log_probs = extract_ssl_and_logprob(self.model, input_values)

        frame2phone = align(log_probs, phone_ids)
        num_phones = len(phone_ids)

        phone_ssl = aggregate_ssl(ssl, frame2phone, num_phones)
        gop = compute_gop(log_probs, phone_ids, frame2phone)
        dur = compute_duration(frame2phone, num_phones)

        return {
            "ssl":       phone_ssl,
            "gop":       gop,
            "dur":       dur,
            "phone_ids": torch.tensor(phone_ids, dtype=torch.long),
            "scores":    torch.tensor(scores, dtype=torch.float),
        }
