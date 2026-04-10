"""
Custom data collator for CTC training.
Pads inputs and labels independently, as required by CTC loss.
"""

from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from transformers import Wav2Vec2Processor


@dataclass
class DataCollatorCTCWithPadding:
    """
    Pads input_values and labels independently — CTC requires this.
    Labels are padded with -100 so they are ignored in loss computation.
    """
    processor:  Wav2Vec2Processor
    padding:    Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]}           for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace pad token id with -100 → ignored by CTC loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
