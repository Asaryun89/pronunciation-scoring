"""
Fine-tune HuBERT-Large (pretrained on LibriSpeech) with CTC
on the SpeechOcean762 dataset for text prediction.

This is Stage 1 of the pronunciation scoring pipeline:
  SSL Pretext (text prediction via CTC)  →  Transfer to scoring heads

Install dependencies:
    pip install transformers datasets jiwer soundfile librosa accelerate

Usage:
    python train_hubert_speechocean.py
"""

import json
import os

import torch
from transformers import Trainer, TrainingArguments

from ctc_training.config import OUTPUT_DIR, SEED, TRAINING_ARGS
from ctc_training.collator import DataCollatorCTCWithPadding
from ctc_training.data import load_speechocean, prepare_dataset
from ctc_training.metrics import make_compute_metrics
from ctc_training.model import build_model
from ctc_training.processor import build_vocab, get_processor


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
        tokenizer        = processor.feature_extractor,
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
