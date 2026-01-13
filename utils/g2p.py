from __future__ import annotations

from typing import List, Tuple


PHONEME_SYMBOLS = ["<pad>"] + [chr(c) for c in range(ord("a"), ord("z") + 1)] + ["sil"]
PHONEME_TO_ID = {sym: idx for idx, sym in enumerate(PHONEME_SYMBOLS)}


def text_to_phonemes(text: str) -> Tuple[List[str], List[int]]:
    """Simple letter-based G2P placeholder for demo purposes."""
    symbols: List[str] = []
    for ch in text.lower():
        if "a" <= ch <= "z":
            symbols.append(ch)
    if not symbols:
        symbols = ["sil"]
    ids = [PHONEME_TO_ID[sym] for sym in symbols]
    return symbols, ids
