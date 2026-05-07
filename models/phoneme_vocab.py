from __future__ import annotations

"""
ARPABET phoneme vocabulary and token normalization for pronunciation scoring.

Provides a fixed vocabulary of ARPABET phonemes with stress markers,
plus helpers to normalize and encode raw phoneme tokens from SpeechOcean762.
"""

import re
from typing import Any, Dict, List

# ─── Phoneme sets ─────────────────────────────────────────────────────────────

ARPABET_VOWELS: List[str] = [
    "AA", "AE", "AH", "AO", "AW",
    "AY", "EH", "ER", "EY", "IH",
    "IY", "OW", "OY", "UH", "UW",
]

ARPABET_CONSONANTS: List[str] = [
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L",
    "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V",
    "W", "Y", "Z", "ZH",
]

SPECIAL_PHONEMES: List[str] = ["<pad>", "<unk>", "<sil>"]

# ─── Vocabulary construction ──────────────────────────────────────────────────


def _build_phoneme_vocab() -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for token in SPECIAL_PHONEMES:
        vocab[token] = len(vocab)
    for vowel in ARPABET_VOWELS:
        for stress in ("0", "1", "2"):
            vocab[f"{vowel}{stress}"] = len(vocab)
    for consonant in ARPABET_CONSONANTS:
        vocab[consonant] = len(vocab)
    return vocab


PHONEME2ID: Dict[str, int] = _build_phoneme_vocab()
ID2PHONEME: Dict[int, str] = {v: k for k, v in PHONEME2ID.items()}

PAD_ID: int = PHONEME2ID["<pad>"]   # 0
UNK_ID: int = PHONEME2ID["<unk>"]   # 1
SIL_ID: int = PHONEME2ID["<sil>"]   # 2
VOCAB_SIZE: int = len(PHONEME2ID)    # 72

# ─── Regex pattern ────────────────────────────────────────────────────────────

_PHONEME_RE = re.compile(r"^([A-Z]+)([012])?$")

# ─── Normalization ────────────────────────────────────────────────────────────


def normalize_phoneme_token(token: Any) -> str:
    """Normalize a raw phoneme token to a canonical PHONEME2ID key.

    Rules applied in order:
    1. Non-string input → ``"<unk>"``.
    2. Empty / whitespace-only / ``SP`` / ``SIL`` / ``<SIL>`` → ``"<sil>"``.
    3. Strip whitespace and uppercase.
    4. Match regex ``^([A-Z]+)([012])?$``.
       - Vowels without stress marker default to stress ``"0"``.
       - Consonants never carry a stress marker.
    5. Unrecognized base form → ``"<unk>"``.

    Args:
        token: Raw phoneme token from a dataset annotation.

    Returns:
        A canonical key guaranteed to exist in :data:`PHONEME2ID`.
    """
    if not isinstance(token, str):
        return "<unk>"

    stripped = token.strip().upper()

    if not stripped or stripped in ("SP", "SIL", "<SIL>"):
        return "<sil>"

    m = _PHONEME_RE.match(stripped)
    if not m:
        return "<unk>"

    base, stress = m.group(1), m.group(2)

    if base in ARPABET_VOWELS:
        canonical = f"{base}{stress if stress else '0'}"
        return canonical if canonical in PHONEME2ID else "<unk>"

    if base in ARPABET_CONSONANTS:
        return base if base in PHONEME2ID else "<unk>"

    return "<unk>"


def phonemes_to_ids(phonemes: List[Any]) -> List[int]:
    """Map a list of raw phoneme tokens to vocabulary IDs.

    Args:
        phonemes: List of raw phoneme tokens (typically strings).

    Returns:
        List of integer vocabulary IDs.
    """
    return [PHONEME2ID[normalize_phoneme_token(p)] for p in phonemes]
