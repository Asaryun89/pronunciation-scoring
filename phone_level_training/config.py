import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATHS = dict(
    dataset_name="mispeech/speechocean762",

    hubert_dir="./hubert_phoneme_ctc",
    processor_dir="./hubert_phoneme_ctc",
    phone2id_path="./phoneme_vocab.json",

    output_dir="./outputs",
    best_model_path="./outputs/best_phone_model.pt",
)

TRAINING_ARGS = dict(
    output_dir=PATHS["output_dir"],

    num_train_epochs=30,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    learning_rate=1e-4,
    weight_decay=1e-5,

    dataloader_num_workers=4,

    logging_steps=50,
    eval_strategy="epoch",

    save_strategy="epoch",
    save_total_limit=2,

    fp16=True,

    seed=42,
)