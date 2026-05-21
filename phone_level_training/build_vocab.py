"""
Generate phoneme_vocab.json from the SpeechOcean762 phoneme labels.

The vocabulary is built by scanning every phone token that appears in the
dataset splits, then sorting them and assigning consecutive integer IDs.
A [PAD] token (id=0) is prepended so padding never collides with a real phone.

Usage (run from repo root):
    python -m phone_level_training.build_vocab
"""
import json
from pathlib import Path


HERE = Path(__file__).parent


def main():
    from datasets import load_dataset

    print("Loading mispeech/speechocean762 ...")
    ds = load_dataset("mispeech/speechocean762")

    phones: set[str] = set()
    for split in ds:
        for sample in ds[split].select_columns(["words"]):
            for word in sample["words"]:
                phones.update(word["phones"])

    # Sort for a deterministic, human-readable mapping.
    # Reserve id=0 for padding so it can never alias a real phoneme.
    phone2id = {"[PAD]": 0}
    for i, p in enumerate(sorted(phones), start=1):
        phone2id[p] = i

    out_path = HERE / "phoneme_vocab.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(phone2id, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(phone2id)} entries (1 PAD + {len(phones)} phones) → {out_path}")
    print("First 10:", dict(list(phone2id.items())[:10]))


if __name__ == "__main__":
    main()
