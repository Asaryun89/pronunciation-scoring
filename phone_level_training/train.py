import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import HubertForCTC, Wav2Vec2Processor
import json

from config import DEVICE, PATHS, TRAINING_ARGS
from .data import PhoneDataset
from .collator import collate_fn
from .model import PhoneModel


def load_resources():
    with open(PATHS["phone2id_path"]) as f:
        phone2id = json.load(f)

    processor = Wav2Vec2Processor.from_pretrained(PATHS["processor_dir"])
    hubert = HubertForCTC.from_pretrained(PATHS["hubert_dir"]).to(DEVICE)

    return hubert, processor, phone2id


def masked_mse(pred, target, mask):
    mask = mask.float()
    return ((pred - target) ** 2 * mask).sum() / mask.sum()


def evaluate(model, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            ssl = batch["ssl"].to(DEVICE)
            gop = batch["gop"].to(DEVICE)
            dur = batch["dur"].to(DEVICE)
            phone_ids = batch["phone_ids"].to(DEVICE)
            scores = batch["scores"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            pred = model(ssl, gop, dur, phone_ids, mask)

            loss = masked_mse(pred, scores, mask)
            total_loss += loss.item()

    return total_loss / len(loader)


def train():
    dataset = load_dataset(PATHS["dataset_name"])
    hubert, processor, phone2id = load_resources()
    train_ds = PhoneDataset(dataset["train"], hubert, processor, phone2id)
    val_ds = PhoneDataset(dataset["validation"], hubert, processor, phone2id)


    train_loader = DataLoader(
        train_ds,
        batch_size=TRAINING_ARGS["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=TRAINING_ARGS["dataloader_num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=TRAINING_ARGS["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=TRAINING_ARGS["dataloader_num_workers"]
    )

    print("Building model...")
    model = PhoneModel(num_phones=len(phone2id)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_ARGS["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=TRAINING_ARGS["use_amp"])
    best_val = float("inf")

    # TRAIN LOOP

    for epoch in range(TRAINING_ARGS["epochs"]):

        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:

            ssl = batch["ssl"].to(DEVICE)
            gop = batch["gop"].to(DEVICE)
            dur = batch["dur"].to(DEVICE)
            phone_ids = batch["phone_ids"].to(DEVICE)
            scores = batch["scores"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=TRAINING_ARGS["use_amp"]):
                pred = model(ssl, gop, dur, phone_ids, mask)
                loss = masked_mse(pred, scores, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train = total_loss / len(train_loader)

        # VALIDATION
        val_loss = evaluate(model, val_loader)

        print(f"[Epoch {epoch}] Train: {avg_train:.4f} | Val: {val_loss:.4f}")

        # SAVE BEST
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), PATHS["best_model_path"])
            print("Saved best model!")

    print("Training complete.")


if __name__ == "__main__":
    train()