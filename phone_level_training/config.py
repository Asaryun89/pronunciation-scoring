import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_HERE = Path(__file__).parent  # always phone_level_training/, regardless of cwd

PATHS = dict(
    dataset_name="mispeech/speechocean762",

    hubert_dir=str(_HERE / "hubert_phoneme_ctc"),
    processor_dir=str(_HERE / "hubert_phoneme_ctc"),
    phone2id_path=str(_HERE / "phoneme_vocab.json"),

    output_dir=str(_HERE / "outputs"),
    best_model_path=str(_HERE / "outputs" / "best_phone_model.pt"),
)

TRAINING_ARGS = dict(
    epochs=30,
    batch_size=4,
    learning_rate=1e-4,
    weight_decay=1e-5,
    dataloader_num_workers=0,  # >0 requires pickling the HuBERT model, which fails on Windows spawn
    use_amp=torch.cuda.is_available(),  # AMP only makes sense on CUDA
    seed=42,
)
