"""
Fine-tune HuBERT-Large (pretrained on LibriSpeech) with CTC
on the SpeechOcean762 dataset for text prediction.

This is Stage 1 of the pronunciation scoring pipeline:
  SSL Pretext (text prediction via CTC)  →  Transfer to scoring heads

Install dependencies:
    pip install transformers datasets jiwer soundfile librosa accelerate

Usage (from the repository root):
    python -m stage1_ctc.train
    # or: python stage1_ctc/train.py
"""

import json
import os
import sys

import torch
from transformers import Trainer, TrainingArguments

# Path fix — support both `python stage1_ctc/train.py` and `python -m stage1_ctc.train`
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from stage1_ctc.config import OUTPUT_DIR, SEED, TRAINING_ARGS      # noqa: E402
from stage1_ctc.collator import DataCollatorCTCWithPadding         # noqa: E402
from stage1_ctc.data import load_speechocean, prepare_dataset      # noqa: E402
from stage1_ctc.metrics import make_compute_metrics                # noqa: E402
from stage1_ctc.model import build_model                           # noqa: E402
from stage1_ctc.processor import build_vocab, get_processor        # noqa: E402


def main():
    torch.manual_seed(SEED)

    # --- Data ---
    dataset = load_speechocean()
    build_vocab(dataset)
    processor = get_processor()
    dataset   = prepare_dataset(dataset, processor)

    # --- Model ---
    model = build_model(processor)

    # --- Collator ---
    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # --- Trainer ---
    training_args = TrainingArguments(**TRAINING_ARGS)

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = dataset["train"],
        eval_dataset     = dataset["validation"],
        processing_class = processor.feature_extractor,
        data_collator    = collator,
        compute_metrics  = make_compute_metrics(processor),
    )

    # --- Train ---
    print("\n── Starting CTC fine-tuning on SpeechOcean762 ──\n")
    trainer.train()

    # --- Save ---
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")

    # --- Final eval on test set ---
    print("\nEvaluating on test set ...")
    test_results = trainer.evaluate(eval_dataset=dataset["test"])
    print(f"Test WER: {test_results['eval_wer']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)


if __name__ == "__main__":
    main()
