"""
Global configuration for HuBERT-CTC fine-tuning on SpeechOcean762.
Edit values here to change model, dataset, paths, or training hyperparameters.
"""

import torch

MODEL_NAME        = "facebook/hubert-large-ls960-ft"   # HuBERT-Large fine-tuned on LibriSpeech 960h
                                                        # Use "facebook/hubert-large-ll60k" for base pretrain only
DATASET_NAME      = "mispeech/speechocean762"
OUTPUT_DIR        = "./outputs/hubert-large-speechocean-ctc"
VOCAB_PATH        = "./outputs/vocab.json"             # built from training transcripts
SAMPLING_RATE     = 16_000
MAX_DURATION_SEC  = 20.0                               # drop utterances longer than this
SEED              = 42

TRAINING_ARGS = dict(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 30,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    gradient_accumulation_steps = 4,                   # effective batch = 16
    learning_rate               = 1e-4,
    warmup_ratio                = 0.1,
    lr_scheduler_type           = "linear",
    weight_decay                = 0.01,
    fp16                        = torch.cuda.is_available(),
    train_sampling_strategy     = "group_by_length",   # speed: batch similar-length seqs together
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    logging_steps               = 50,
    save_total_limit            = 3,
    report_to                   = "none",              # set "wandb" if you use W&B
    dataloader_num_workers      = 4,
    seed                        = SEED,
    # Gradient checkpointing — saves VRAM at cost of ~20% speed
    gradient_checkpointing      = True,
)
