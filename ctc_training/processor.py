"""
Vocabulary building and Wav2Vec2Processor construction for CTC training.
"""

import json
import os
import re

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from .config import SAMPLING_RATE

SPECIAL_CHARS_RE = re.compile(r"[^a-z\s']")   # keep lowercase letters, space, apostrophe


def normalise_text(text: str) -> str:
    text = text.lower().strip()
    text = SPECIAL_CHARS_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text


def build_vocab(dataset, vocab_path: str = "vocab.json"):
    """Extract character vocabulary from the training transcripts."""
    if os.path.exists(vocab_path):
        print(f"Vocab already exists at {vocab_path}, skipping build.")
        return

    all_text = " ".join(
        normalise_text(sample["text"])
        for sample in dataset["train"]
    )
    vocab_chars = sorted(set(all_text))
    vocab_dict = {ch: idx for idx, ch in enumerate(vocab_chars)}

    # Required CTC special tokens
    vocab_dict["|"] = len(vocab_dict)   # word boundary (space replacement)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f, indent=2)
    print(f"Vocab size: {len(vocab_dict)}  →  saved to {vocab_path}")


def get_processor(vocab_path: str = "vocab.json") -> Wav2Vec2Processor:
    """Build a Wav2Vec2Processor from a vocab file."""
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size        = 1,
        sampling_rate       = SAMPLING_RATE,
        padding_value       = 0.0,
        do_normalize        = True,
        return_attention_mask = True,
    )
    return Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
