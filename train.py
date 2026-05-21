import torch

# torchvision is installed but version-mismatched with torch — its __init__
# crashes while registering fake ops for ops that don't exist in this torch build.
# Patch register_fake to swallow those errors, import torchvision, then restore.
_orig_register_fake = torch.library.register_fake
def _guarded_register_fake(op, *args, **kwargs):
    orig_dec = _orig_register_fake(op, *args, **kwargs)
    def decorator(fn):
        try:
            return orig_dec(fn)
        except RuntimeError:
            return fn
    return decorator
torch.library.register_fake = _guarded_register_fake
try:
    import torchvision  # noqa: F401
finally:
    torch.library.register_fake = _orig_register_fake
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers.models.hubert.modeling_hubert import HubertForCTC
import json
import os

from phone_level_training.config import DEVICE, PATHS, TRAINING_ARGS
from phone_level_training.data import PhoneDataset
from phone_level_training.collator import collate_fn
from phone_level_training.model import PhoneModel


def load_resources():
    with open(PATHS["phone2id_path"]) as f:
        phone2id = json.load(f)

    hubert = HubertForCTC.from_pretrained(
        PATHS["hubert_dir"],
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    hubert.eval()

    return hubert, phone2id


def masked_mse(pred, target, mask):
    mask = mask.float()
    return ((pred - target) ** 2 * mask).sum() / mask.sum()


def evaluate(model, loader):
    """Returns dict with mse, mae, pearson over all valid (non-padded) tokens."""
    model.eval()
    all_pred, all_target = [], []

    with torch.no_grad():
        for batch in loader:
            ssl = batch["ssl"].to(DEVICE)
            gop = batch["gop"].to(DEVICE)
            dur = batch["dur"].to(DEVICE)
            phone_ids = batch["phone_ids"].to(DEVICE)
            scores = batch["scores"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            pred = model(ssl, gop, dur, phone_ids, mask)

            # collect only valid tokens
            all_pred.append(pred[mask].cpu())
            all_target.append(scores[mask].cpu())

    p = torch.cat(all_pred).numpy()
    t = torch.cat(all_target).numpy()

    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))

    if p.std() > 1e-8 and t.std() > 1e-8:
        pearson = float(np.corrcoef(p, t)[0, 1])
    else:
        pearson = 0.0

    return {"mse": mse, "mae": mae, "pearson": pearson}


def train(
    use_ssl: bool = True,
    use_gop: bool = True,
    use_dur: bool = True,
    use_phone_embed: bool = True,
    tag: str = "all",
):
    """Train one ablation configuration.

    Args:
        use_ssl / use_gop / use_dur / use_phone_embed: feature ablation flags.
        tag: short label used for checkpoint naming.

    Returns:
        dict with best val metrics.
    """
    torch.manual_seed(TRAINING_ARGS["seed"])

    dataset = load_dataset(PATHS["dataset_name"])
    hubert, phone2id = load_resources()
    num_hubert_layers = hubert.config.num_hidden_layers  # 12 for base, 24 for large
    ssl_dim = hubert.config.hidden_size                  # 768 for base, 1024 for large

    train_ds = PhoneDataset(dataset["train"], hubert, phone2id)
    val_ds = PhoneDataset(dataset["test"], hubert, phone2id)

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAINING_ARGS["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=TRAINING_ARGS["dataloader_num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=TRAINING_ARGS["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=TRAINING_ARGS["dataloader_num_workers"],
    )

    model = PhoneModel(
        num_phones=len(phone2id),
        ssl_dim=ssl_dim,
        num_hubert_layers=num_hubert_layers,
        use_ssl=use_ssl,
        use_gop=use_gop,
        use_dur=use_dur,
        use_phone_embed=use_phone_embed,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_ARGS["learning_rate"],
        weight_decay=TRAINING_ARGS["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=TRAINING_ARGS["use_amp"])

    os.makedirs(PATHS["output_dir"], exist_ok=True)
    ckpt_path = os.path.join(PATHS["output_dir"], f"best_phone_model_{tag}.pt")

    best_val_mse = float("inf")
    best_metrics = {}

    for epoch in range(TRAINING_ARGS["epochs"]):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{tag}] Epoch {epoch}")

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
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader)

        print(
            f"[{tag}] Epoch {epoch:02d}  "
            f"train_mse={avg_train:.4f}  "
            f"val_mse={val_metrics['mse']:.4f}  "
            f"val_mae={val_metrics['mae']:.4f}  "
            f"pearson={val_metrics['pearson']:.4f}"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_metrics = val_metrics
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best checkpoint to {ckpt_path}")

    return best_metrics


if __name__ == "__main__":
    results = train()
    print("\nBest val metrics:", results)
