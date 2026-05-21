import argparse
import json
from .predictor import PronunciationPredictor, PredictorConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav",        required=True, help="Path to WAV file (any sample rate; will be resampled to 16kHz)")
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--whisper_device", default="cpu")
    ap.add_argument("--lang",       default="en")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to trained HubertMultiTask checkpoint (.pt). "
                         "Omit to run with random weights (scores not meaningful).")
    args = ap.parse_args()

    cfg = PredictorConfig(
        device=args.device,
        whisper_device=args.whisper_device,
        checkpoint_path=args.checkpoint,
    )
    pred = PronunciationPredictor(cfg)
    out = pred.predict(args.wav, language=args.lang)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()