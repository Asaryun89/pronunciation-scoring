# Fine-tune HuBERT on SpeechOcean762 for utterance-level pronunciation scoring.
# Produces sentence-level scores only (total, accuracy, fluency, prosodic, completeness).
# Word and phone prediction heads are intentionally omitted — to be trained separately.
# pip install torch torchaudio transformers datasets accelerate scipy

import csv
import os
import sys
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from scipy.stats import pearsonr, spearmanr

from datasets import load_dataset, Audio
from transformers import AutoTokenizer

try:
    from accelerate import Accelerator  # type: ignore
except Exception:
    Accelerator = None

# ---------------------------------------------------------------------------
# Path fix — support both `python models/train.py` and `python -m models.train`
# ---------------------------------------------------------------------------
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from models.constants import SENT_DIMS, PROSODY_DIMS, ACTIVE_SENT_IDXS  # noqa: E402
from models.hubert_multitask import HubertMultiTask                             # noqa: E402
from data.collate import Collator                                                # noqa: E402
from data.validate import validate_example_schema                               # noqa: E402


# ---------------------------------------------------------------------------
# Accelerator shim — single-process fallback when `accelerate` is not installed
# ---------------------------------------------------------------------------
class SimpleAccelerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_main_process = True

    def prepare(self, *args):
        prepared = []
        for obj in args:
            if isinstance(obj, nn.Module):
                prepared.append(obj.to(self.device))
            elif obj is None:
                prepared.append(None)
            else:
                prepared.append(obj)
        return tuple(prepared)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model


def create_accelerator():
    if Accelerator is None:
        print("`accelerate` not found. Falling back to single-process training.")
        return SimpleAccelerator()
    return Accelerator()


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0 or step >= warmup_steps:
            return 1.0
        return float(step) / float(max(1, warmup_steps))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch:   Dict[str, Any],
    w_sent:  float,
    w_pfeat: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = outputs["sent_pred"].device

    sent_targets  = batch["sent_targets"].to(device)
    prosody_feats = batch["prosody_feats"].to(device)

    # Only train on active sentence dims — completeness (idx 4) is excluded because
    # SpeechOcean learners nearly always score 10/10, collapsing the head to a constant.
    # MSE loss matches the paper (Kim et al., 2022).
    sent_loss  = F.mse_loss(
        outputs["sent_pred"][:, ACTIVE_SENT_IDXS],
        sent_targets[:, ACTIVE_SENT_IDXS],
    )
    pfeat_loss = F.mse_loss(outputs["prosody_pred"], prosody_feats)

    total_loss = w_sent * sent_loss + w_pfeat * pfeat_loss

    logs = {
        "loss_total":   float(total_loss.detach().cpu()),
        "loss_sent":    float(sent_loss.detach().cpu()),
        "loss_prosody": float(pfeat_loss.detach().cpu()),
    }
    return total_loss, logs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    accelerator: Any,
    w_sent:      float = 1.0,
    w_pfeat:     float = 0.0,
) -> Dict[str, float]:
    model.eval()

    all_pred:   List[torch.Tensor] = []
    all_target: List[torch.Tensor] = []
    total_loss = 0.0
    n_steps    = 0

    for batch in loader:
        batch   = move_batch_to_device(batch, accelerator.device)
        outputs = model(
            input_values=batch["input_values"],
            attention_mask=batch["attention_mask"],
            text_input_ids=batch.get("text_input_ids"),
            text_attention_mask=batch.get("text_attention_mask"),
        )
        loss, _ = compute_loss(outputs, batch, w_sent, w_pfeat)
        total_loss += float(accelerator.gather(loss.detach()).mean().cpu())
        n_steps    += 1

        all_pred.append(outputs["sent_pred"].detach().cpu())
        all_target.append(batch["sent_targets"].cpu())

    metrics: Dict[str, float] = {"val_loss": total_loss / max(1, n_steps)}

    if all_pred:
        preds   = torch.cat(all_pred,   dim=0).numpy()  # (N, 5)
        targets = torch.cat(all_target, dim=0).numpy()  # (N, 5)

        for i, dim in enumerate(SENT_DIMS):
            p, t = preds[:, i], targets[:, i]
            metrics[f"mae_{dim}"]  = float(np.mean(np.abs(p - t)))
            metrics[f"rmse_{dim}"] = float(np.sqrt(np.mean((p - t) ** 2)))
            if len(p) > 1 and np.std(p) > 1e-6 and np.std(t) > 1e-6:
                metrics[f"pearson_{dim}"]  = float(pearsonr(p, t)[0])
                metrics[f"spearman_{dim}"] = float(spearmanr(p, t)[0])
            else:
                metrics[f"pearson_{dim}"]  = 0.0
                metrics[f"spearman_{dim}"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune HuBERT on SpeechOcean762")
    ap.add_argument("--dataset",     type=str,   default="mispeech/speechocean762")
    ap.add_argument("--train_split", type=str,   default="train")
    ap.add_argument("--valid_split", type=str,   default="test")
    ap.add_argument("--model",       type=str,   default="facebook/hubert-base-ls960")
    ap.add_argument("--out_dir",     type=str,   default="ckpt_hubert_multitask")
    ap.add_argument("--epochs",      type=int,   default=5)
    ap.add_argument("--batch_size",  type=int,   default=4)
    ap.add_argument("--lr",          type=float, default=2e-5)
    ap.add_argument("--wd",          type=float, default=0.01)
    ap.add_argument("--dropout",     type=float, default=0.1)
    ap.add_argument("--freeze_fe",   action="store_true",
                    help="Freeze HuBERT CNN feature extractor weights")
    ap.add_argument("--d_model",     type=int,   default=256,
                    help="Shared projection dimension for audio and text streams")
    ap.add_argument("--num_heads",   type=int,   default=8,
                    help="Attention heads in cross-attention and Transformer block")
    ap.add_argument("--num_audio_transformer_layers", type=int, default=1,
                    help="Pre-fusion audio self-attention depth (0 = disabled)")
    ap.add_argument("--num_transformer_layers", type=int, default=2,
                    help="Number of post-fusion Transformer encoder layers")
    ap.add_argument("--mlp_hidden_layers",      type=int, default=2,
                    help="Hidden FC→ReLU→Dropout blocks in the MLP scoring head")
    ap.add_argument("--num_unfreeze_hubert_layers", type=int, default=12,
                    help="Unfreeze top-N HuBERT transformer layers (12 = full backbone, 0 = all frozen)")
    ap.add_argument("--num_workers", type=int,   default=2)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--w_sent",      type=float, default=1.0)
    ap.add_argument("--w_pfeat",     type=float, default=0.0,
                    help="Weight for auxiliary prosody feature loss (0 = disabled)")
    ap.add_argument("--warmup_steps", type=int,   default=100,
                    help="Linear LR warmup steps (0 = disabled)")
    ap.add_argument("--patience",    type=int,   default=3,
                    help="Early stopping patience on pearson_total (0 = disabled)")
    ap.add_argument("--log_every",   type=int,   default=50)
    ap.add_argument("--text_model", type=str,   default="Qwen/Qwen3-Embedding-0.6B",
                    help="HuggingFace model for text embedding stream ('' to disable)")
    ap.add_argument("--no_freeze_text",  action="store_true",
                    help="Fine-tune the text encoder instead of keeping it frozen")
    args = ap.parse_args()

    set_seed(args.seed)
    accelerator = create_accelerator()

    # ---- Dataset ----
    ds       = load_dataset(args.dataset)
    ds       = ds.cast_column("audio", Audio(decode=False))
    train_ds = ds[args.train_split]
    valid_ds = ds.get(args.valid_split)

    for i in range(min(32, len(train_ds))):
        validate_example_schema(train_ds[i], args.train_split, i)
    if valid_ds is not None:
        for i in range(min(32, len(valid_ds))):
            validate_example_schema(valid_ds[i], args.valid_split, i)

    # ---- Collators + loaders ----
    text_tokenizer  = AutoTokenizer.from_pretrained(args.text_model) if args.text_model else None
    collator_kwargs = dict(
        sample_rate=16000,
        text_tokenizer=text_tokenizer,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=Collator(**collator_kwargs, split_name=args.train_split),
    )
    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=Collator(**collator_kwargs, split_name=args.valid_split),
        )

    # ---- Model + optimiser ----
    model = HubertMultiTask(
        model_name=args.model,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_audio_transformer_layers=args.num_audio_transformer_layers,
        num_transformer_layers=args.num_transformer_layers,
        mlp_hidden_layers=args.mlp_hidden_layers,
        dropout=args.dropout,
        freeze_fe=args.freeze_fe,
        num_unfreeze_hubert_layers=args.num_unfreeze_hubert_layers,
        text_model_name=args.text_model or None,
        freeze_text_encoder=not args.no_freeze_text,
        prosody_feat_dim=len(PROSODY_DIMS),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = make_warmup_scheduler(optimizer, args.warmup_steps)

    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_pearson_total = -2.0
    epochs_no_improve  = 0

    # ---- CSV log setup ----
    _TRAIN_COLS = ["epoch", "step", "phase", "train_loss", "loss_sent", "loss_prosody"]
    _VAL_COLS   = ["val_loss"] + [
        f"{metric}_{dim}"
        for metric in ("mae", "rmse", "pearson", "spearman")
        for dim in SENT_DIMS
    ]
    _CSV_COLS = _TRAIN_COLS + _VAL_COLS
    csv_path  = os.path.join(args.out_dir, "result.csv")

    csv_file   = open(csv_path, "w", newline="") if accelerator.is_main_process else None
    csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLS, restval="") if csv_file else None
    if csv_writer:
        csv_writer.writeheader()

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss   = 0.0
        running_steps  = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, accelerator.device)
            outputs = model(
                input_values=batch["input_values"],
                attention_mask=batch["attention_mask"],
                text_input_ids=batch.get("text_input_ids"),
                text_attention_mask=batch.get("text_attention_mask"),
            )
            loss, logs = compute_loss(outputs, batch, args.w_sent, args.w_pfeat)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss  += logs["loss_total"]
            running_steps += 1
            if accelerator.is_main_process and step % args.log_every == 0:
                avg_loss = running_loss / running_steps
                extra = ""
                if args.w_pfeat > 0:
                    extra += f"  prosody={logs['loss_prosody']:.4f}"
                print(
                    f"[epoch {epoch} step {step}] "
                    f"train_loss={avg_loss:.4f}  "
                    f"sent={logs['loss_sent']:.4f}"
                    f"{extra}"
                )
                if csv_writer:
                    csv_writer.writerow({
                        "epoch":        epoch,
                        "step":         step,
                        "phase":        "train",
                        "train_loss":   f"{avg_loss:.6f}",
                        "loss_sent":    f"{logs['loss_sent']:.6f}",
                        "loss_prosody": f"{logs['loss_prosody']:.6f}",
                    })
                    csv_file.flush()
                running_loss  = 0.0
                running_steps = 0

        # ---- Validation ----
        if valid_loader is not None:
            metrics = eval_epoch(model, valid_loader, accelerator, args.w_sent, args.w_pfeat)
            if accelerator.is_main_process:
                print(f"\n== epoch {epoch} validation ==")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.4f}")

                if csv_writer:
                    row = {"epoch": epoch, "step": "end", "phase": "val"}
                    row.update({k: f"{v:.6f}" for k, v in metrics.items()})
                    csv_writer.writerow(row)
                    csv_file.flush()

                pearson_total = metrics.get("pearson_total", -2.0)
                if pearson_total > best_pearson_total:
                    best_pearson_total = pearson_total
                    epochs_no_improve  = 0
                    ckpt_path = os.path.join(args.out_dir, "best.pt")
                    torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
                    print(f"  -> saved best checkpoint (pearson_total={pearson_total:.4f})")
                else:
                    epochs_no_improve += 1
                    if args.patience > 0 and epochs_no_improve >= args.patience:
                        print(f"  -> early stopping (no improvement for {args.patience} epochs)")
                        break
        else:
            if accelerator.is_main_process:
                ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch}.pt")
                torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
                print(f"Saved {ckpt_path}")

    if csv_file:
        csv_file.close()


if __name__ == "__main__":
    main()
