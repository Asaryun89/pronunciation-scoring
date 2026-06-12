"""
Dataset loading and preprocessing for SpeechOcean762 CTC training.
"""

import numpy as np
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor

from .config import DATASET_NAME, SAMPLING_RATE, MAX_DURATION_SEC, SEED
from .processor import normalise_text


def load_speechocean():
    """
    Load SpeechOcean762 from HuggingFace.
    Splits: train / test — we carve a 10% validation set from train.
    """
    print("Loading SpeechOcean762 ...")
    dataset = load_dataset(DATASET_NAME)

    # Cast audio column to the target sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    # Train / val split
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=SEED)
    dataset["train"] = train_val["train"]
    dataset["validation"] = train_val["test"]

    print(f"  Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")
    return dataset


def make_preprocess_fn(processor: Wav2Vec2Processor):
    """Return a per-sample preprocessing function compatible with dataset.map()."""
    max_samples = int(MAX_DURATION_SEC * SAMPLING_RATE)

    def preprocess(batch):
        audio = batch["audio"]
        waveform = np.array(audio["array"], dtype=np.float32)

        # Drop if too long
        if len(waveform) > max_samples:
            return {
                "input_values": None,
                "attention_mask": None,
                "labels": None,
            }

        inputs = processor(
            waveform,
            sampling_rate=SAMPLING_RATE,
            return_attention_mask=True,
        )

        text = normalise_text(batch["text"])
        labels = processor.tokenizer(text).input_ids

        return {
            "input_values":  inputs.input_values[0],
            "attention_mask": inputs.attention_mask[0],
            "labels":        labels,
        }

    return preprocess


def prepare_dataset(dataset, processor: Wav2Vec2Processor):
    """Apply preprocessing and filter out too-long utterances."""
    preprocess_fn = make_preprocess_fn(processor)

    cols_to_remove = [
        c for c in dataset["train"].column_names
        if c not in {"input_values", "attention_mask", "labels"}
    ]

    dataset = dataset.map(
        preprocess_fn,
        remove_columns=cols_to_remove,
        num_proc=4,
        desc="Preprocessing audio",
    )

    # Remove rows that were too long (None values)
    dataset = dataset.filter(lambda x: x["input_values"] is not None)
    return dataset
