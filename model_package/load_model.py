"""
Minimal example: load a trained HubertMultiTask checkpoint and score an audio file.

Usage:
    python load_model.py --wav path/to/audio.wav --checkpoint path/to/best.pt

The checkpoint (best.pt) must be placed/downloaded separately.
HuggingFace models are downloaded automatically on first run:
  - facebook/hubert-base-ls960  (~360 MB)
  - Qwen/Qwen3-Embedding-0.6B   (~600 MB)
  - Faster-Whisper small        (~460 MB)
"""
import argparse
import json
from .inference.predictor import PronunciationPredictor, PredictorConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = PredictorConfig(
        device=args.device,
        checkpoint_path=args.checkpoint,
    )
    predictor = PronunciationPredictor(cfg)
    result = predictor.predict(args.wav, language=args.lang)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
