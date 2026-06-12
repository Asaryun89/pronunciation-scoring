# Schema validation for SpeechOcean762 rows, run on a sample of each split
# before training starts (see stage2_scoring/train.py). Catches dataset/loader drift
# early instead of failing mid-epoch inside the collator.
#
# Canonical sample shape (see AGENTS.md "Data Representation"):
#   utterance: accuracy, completeness, fluency, prosodic, total, text,
#              speaker, gender, age
#   words[]:   text, accuracy, stress, total, phones, phones-accuracy,
#              mispronunciations
#   audio:     {path, bytes}  (cast with Audio(decode=False))

from numbers import Number
from typing import Any, Dict

_UTT_NUMERIC_FIELDS = ("accuracy", "completeness", "fluency", "prosodic", "total")
_WORD_NUMERIC_FIELDS = ("accuracy", "stress", "total")


def _fail(ctx: str, msg: str) -> None:
    raise ValueError(f"dataset schema error at {ctx}: {msg}")


def validate_example_schema(example: Dict[str, Any], split_name: str, idx: int) -> None:
    """Raise ValueError if `example` does not match the expected SpeechOcean762 schema."""
    ctx = f"{split_name}[{idx}]"

    # ---- utterance-level labels ----
    for field in _UTT_NUMERIC_FIELDS:
        if field not in example:
            _fail(ctx, f"missing utterance field '{field}'")
        if not isinstance(example[field], Number):
            _fail(ctx, f"utterance field '{field}' is {type(example[field]).__name__}, expected number")

    if not isinstance(example.get("text"), str) or not example["text"].strip():
        _fail(ctx, "missing or empty 'text'")

    # ---- word annotations ----
    words = example.get("words")
    if not isinstance(words, list):
        _fail(ctx, f"'words' is {type(words).__name__}, expected list")
    for w_idx, word in enumerate(words):
        wctx = f"{ctx}.words[{w_idx}]"
        if not isinstance(word, dict):
            _fail(wctx, f"expected dict, got {type(word).__name__}")
        if not isinstance(word.get("text"), str):
            _fail(wctx, "missing or non-string 'text'")
        for field in _WORD_NUMERIC_FIELDS:
            if not isinstance(word.get(field), Number):
                _fail(wctx, f"missing or non-numeric '{field}'")
        phones     = word.get("phones")
        phones_acc = word.get("phones-accuracy")
        if not isinstance(phones, list) or not isinstance(phones_acc, list):
            _fail(wctx, "'phones' and 'phones-accuracy' must be lists")
        if len(phones) != len(phones_acc):
            _fail(wctx, f"phones ({len(phones)}) and phones-accuracy ({len(phones_acc)}) length mismatch")

    # ---- audio payload ----
    audio = example.get("audio")
    if not isinstance(audio, dict):
        _fail(ctx, f"'audio' is {type(audio).__name__}, expected dict — "
                   "cast the column with Audio(decode=False) before training")
    if not audio.get("bytes") and not audio.get("path"):
        _fail(ctx, "audio payload has neither 'bytes' nor 'path'")
