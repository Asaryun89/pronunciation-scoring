"""
Evaluation metric (Word Error Rate) for CTC training.
"""

import numpy as np
from jiwer import wer
from transformers import Wav2Vec2Processor


def make_compute_metrics(processor: Wav2Vec2Processor):
    """Return a compute_metrics function compatible with HuggingFace Trainer."""
    def compute_metrics(pred):
        pred_logits  = pred.predictions
        pred_ids     = np.argmax(pred_logits, axis=-1)
        label_ids    = pred.label_ids

        # Replace -100 back to pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        error_rate = wer(label_str, pred_str)
        return {"wer": error_rate}

    return compute_metrics
