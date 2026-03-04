import argparse
import json
from .predictor import PronunciationPredictor, PredictorConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Path to 16kHz WAV (or any WAV; it will be resampled)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--whisper_device", default="cpu")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    cfg = PredictorConfig(
        device=args.device,
        whisper_device=args.whisper_device,
    )
    pred = PronunciationPredictor(cfg)
    out = pred.predict(args.wav, language=args.lang)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()