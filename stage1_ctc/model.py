"""
HuBERT-CTC model construction.
"""

from transformers import HubertForCTC, Wav2Vec2Processor

from .config import MODEL_NAME


def build_model(processor: Wav2Vec2Processor) -> HubertForCTC:
    """
    Load HuBERT-Large pretrained on LibriSpeech and attach a fresh CTC head.
    The CNN feature encoder is frozen; only transformer layers are fine-tuned.
    Remove model.freeze_feature_encoder() for full fine-tuning.
    """
    model = HubertForCTC.from_pretrained(
        MODEL_NAME,
        ctc_loss_reduction          = "mean",
        pad_token_id                = processor.tokenizer.pad_token_id,
        vocab_size                  = len(processor.tokenizer),
        ignore_mismatched_sizes     = True,   # lm_head will be re-initialised
    )

    # Freeze the feature encoder (CNN layers) — only fine-tune transformer layers
    model.freeze_feature_encoder()

    return model
