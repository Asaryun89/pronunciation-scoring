# train_hubert_speechocean_hf.py
# pip install torch torchaudio transformers datasets accelerate

import os
import math
import io
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import soundfile as sf
from scipy.signal import resample_poly

from datasets import load_dataset, Audio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
try:
    from accelerate import Accelerator  # type: ignore
except Exception:
    Accelerator = None


class SimpleAccelerator:
    """
    Minimal fallback used when `accelerate` is not installed.
    Supports single-process CPU/GPU training.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_main_process = True

    def prepare(self, *args):
        prepared = []
        for obj in args:
            if isinstance(obj, nn.Module):
                prepared.append(obj.to(self.device))
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
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


# -------------------------
# Helpers
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def strict_float(x: Any, field_name: str, row_hint: str) -> float:
    try:
        v = float(x)
    except Exception as exc:
        raise ValueError(f"{row_hint}: field `{field_name}` must be numeric, got {type(x).__name__}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{row_hint}: field `{field_name}` must be finite, got {v}")
    return v


def require_field(obj: Dict[str, Any], key: str, row_hint: str) -> Any:
    if key not in obj:
        raise ValueError(f"{row_hint}: missing required field `{key}`")
    return obj[key]


def build_row_hint(example: Dict[str, Any], split_name: str, index: Any) -> str:
    audio = example.get("audio", {}) if isinstance(example.get("audio", {}), dict) else {}
    audio_path = audio.get("path", "unknown_path")
    speaker = example.get("speaker", "unknown_speaker")
    text = str(example.get("text", ""))[:40]
    return f"[{split_name} idx={index} speaker={speaker} audio={audio_path} text={text!r}]"


def validate_example_schema(example: Dict[str, Any], split_name: str, index: Any) -> None:
    row_hint = build_row_hint(example, split_name, index)

    # Required utterance-level fields from AGENTS.md schema
    strict_float(require_field(example, "accuracy", row_hint), "accuracy", row_hint)
    strict_float(require_field(example, "completeness", row_hint), "completeness", row_hint)
    strict_float(require_field(example, "fluency", row_hint), "fluency", row_hint)
    strict_float(require_field(example, "prosodic", row_hint), "prosodic", row_hint)
    strict_float(require_field(example, "total", row_hint), "total", row_hint)
    require_field(example, "text", row_hint)
    require_field(example, "speaker", row_hint)
    require_field(example, "gender", row_hint)
    strict_float(require_field(example, "age", row_hint), "age", row_hint)

    audio = require_field(example, "audio", row_hint)
    if not isinstance(audio, dict):
        raise ValueError(f"{row_hint}: field `audio` must be a dict")
    require_field(audio, "path", row_hint)
    if ("array" not in audio) and ("bytes" not in audio):
        raise ValueError(f"{row_hint}: audio must include either `array` or `bytes`")

    words = require_field(example, "words", row_hint)
    if not isinstance(words, list):
        raise ValueError(f"{row_hint}: field `words` must be a list")
    for wi, w in enumerate(words):
        if not isinstance(w, dict):
            raise ValueError(f"{row_hint}: words[{wi}] must be a dict")
        strict_float(require_field(w, "accuracy", row_hint), f"words[{wi}].accuracy", row_hint)
        strict_float(require_field(w, "stress", row_hint), f"words[{wi}].stress", row_hint)
        strict_float(require_field(w, "total", row_hint), f"words[{wi}].total", row_hint)
        require_field(w, "text", row_hint)
        phones = require_field(w, "phones", row_hint)
        phone_acc = require_field(w, "phones-accuracy", row_hint)
        require_field(w, "mispronunciations", row_hint)
        if not isinstance(phones, list):
            raise ValueError(f"{row_hint}: words[{wi}].phones must be a list")
        if not isinstance(phone_acc, list):
            raise ValueError(f"{row_hint}: words[{wi}].phones-accuracy must be a list")
        for pi, v in enumerate(phone_acc):
            strict_float(v, f"words[{wi}].phones-accuracy[{pi}]", row_hint)


def build_prosody_features(example: Dict[str, Any], row_hint: str) -> List[float]:
    """
    Your dataset does NOT provide `prosody_features` explicitly.
    So we derive a simple vector from existing annotation fields.

    Vector (K=4):
      0) prosodic (sentence)
      1) avg word stress
      2) avg phone accuracy (across all phones)
      3) std phone accuracy
    """
    prosodic = strict_float(example.get("prosodic"), "prosodic", row_hint)
    words = example.get("words", [])

    stresses = []
    phone_accs = []
    for w in words:
        stresses.append(strict_float(w.get("stress"), "words[].stress", row_hint))
        pa = w.get("phones-accuracy", []) or []
        for v in pa:
            phone_accs.append(strict_float(v, "words[].phones-accuracy[]", row_hint))

    if len(stresses) == 0:
        avg_stress = 0.0
    else:
        avg_stress = sum(stresses) / len(stresses)

    if len(phone_accs) == 0:
        mean_pa = 0.0
        std_pa = 0.0
    else:
        mean_pa = sum(phone_accs) / len(phone_accs)
        var = sum((v - mean_pa) ** 2 for v in phone_accs) / max(1, len(phone_accs))
        std_pa = math.sqrt(var)

    return [prosodic, avg_stress, mean_pa, std_pa]


def _to_mono(audio):
    if hasattr(audio, "ndim") and audio.ndim == 2:
        return audio.mean(axis=1)
    return audio


def decode_audio_from_example(example: Dict[str, Any], row_hint: str, target_sr: int = 16000):
    audio = example.get("audio")
    if not isinstance(audio, dict):
        raise ValueError(f"{row_hint}: field `audio` must be a dict")

    if "array" in audio and audio["array"] is not None:
        arr = _to_mono(audio["array"])
        sr = int(audio.get("sampling_rate", target_sr))
    else:
        a_bytes = audio.get("bytes", None)
        a_path = audio.get("path", None)
        if a_bytes is not None:
            arr, sr = sf.read(io.BytesIO(a_bytes), dtype="float32")
        elif a_path:
            arr, sr = sf.read(a_path, dtype="float32")
        else:
            raise ValueError(f"{row_hint}: audio needs either `array`, `bytes`, or `path`")
        arr = _to_mono(arr)

    if sr != target_sr:
        arr = resample_poly(arr, target_sr, sr)
    return arr


# -------------------------
# Collator
# -------------------------
@dataclass
class Collator:
    feature_extractor: Wav2Vec2FeatureExtractor
    sample_rate: int = 16000
    max_words: int = 60
    split_name: str = "unknown"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Validate schema and extract audio arrays with row-level hints.
        audios = []
        for i, ex in enumerate(batch):
            row_ref = f"batch_pos:{i}"
            validate_example_schema(ex, self.split_name, row_ref)
            row_hint = build_row_hint(ex, self.split_name, row_ref)
            arr = decode_audio_from_example(ex, row_hint, target_sr=self.sample_rate)
            audios.append(arr)

        feats = self.feature_extractor(
            audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = feats["input_values"]               # (B, Tsamples)
        attention_mask = feats.get("attention_mask", None) # (B, Tsamples) or None

        # sentence targets: overall(total), accuracy, fluency, prosody(prosodic), completeness
        sent_targets = []
        prosody_feats = []

        # word targets: words[i]["total"]
        Wmax = min(
            self.max_words,
            max(len(ex.get("words", []) or []) for ex in batch) if len(batch) else 1,
        )
        word_scores = torch.zeros(len(batch), Wmax, dtype=torch.float32)
        word_mask = torch.zeros(len(batch), Wmax, dtype=torch.bool)

        for i, ex in enumerate(batch):
            row_ref = f"batch_pos:{i}"
            row_hint = build_row_hint(ex, self.split_name, row_ref)
            total = strict_float(ex.get("total"), "total", row_hint)
            acc = strict_float(ex.get("accuracy"), "accuracy", row_hint)
            flu = strict_float(ex.get("fluency"), "fluency", row_hint)
            pro = strict_float(ex.get("prosodic"), "prosodic", row_hint)
            comp = strict_float(ex.get("completeness"), "completeness", row_hint)
            sent_targets.append([total, acc, flu, pro, comp])

            pf = build_prosody_features(ex, row_hint)  # K=4
            prosody_feats.append(pf)

            ws = ex.get("words", []) or []
            ws = ws[:Wmax]
            if len(ws) > 0:
                scores = [strict_float(w.get("total"), f"words[{j}].total", row_hint) for j, w in enumerate(ws)]
                word_scores[i, :len(scores)] = torch.tensor(scores, dtype=torch.float32)
                word_mask[i, :len(scores)] = True

        sent_targets = torch.tensor(sent_targets, dtype=torch.float32)      # (B,5)
        prosody_feats = torch.tensor(prosody_feats, dtype=torch.float32)    # (B,K=4)

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "sent_targets": sent_targets,
            "prosody_feats": prosody_feats,
            "word_scores": word_scores,
            "word_mask": word_mask,
        }


# -------------------------
# Model
# -------------------------
class HubertMultiTask(nn.Module):
    """
    Heads:
      - sentence_head: 5 scalars (total, accuracy, fluency, prosodic, completeness)
      - prosody_feat_head: K=4 derived features
      - word_head: per-word scalar, using EVEN frame segmentation baseline
    """
    def __init__(self, model_name: str, prosody_feat_dim: int = 4, dropout: float = 0.1, freeze_fe: bool = True):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name)
        if freeze_fe:
            self.hubert.feature_extractor._freeze_parameters()

        hidden = self.hubert.config.hidden_size

        self.sentence_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 5),
        )

        self.prosody_feat_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, prosody_feat_dim),
        )

        self.word_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        word_mask: torch.Tensor,  # (B, W)
    ) -> Dict[str, torch.Tensor]:
        out = self.hubert(input_values=input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, Tfrm, H)

        B, Tfrm, H = hidden.shape

        # Sentence embedding: mean over frames (HuBERT already outputs frames)
        sent_emb = hidden.mean(dim=1)  # (B,H)

        sent_pred = self.sentence_head(sent_emb)          # (B,5)
        prosody_feat_pred = self.prosody_feat_head(sent_emb)  # (B,K)

        # Word-level baseline: even segmentation across Tfrm into W segments
        W = word_mask.size(1)
        word_pred = torch.zeros(B, W, device=hidden.device, dtype=torch.float32)

        for b in range(B):
            valid_w = int(word_mask[b].sum().item())
            if valid_w == 0:
                continue
            # split frames into valid_w bins
            # boundaries: [0..Tfrm] into valid_w equal parts
            for wi in range(valid_w):
                s = int(round(wi * Tfrm / valid_w))
                e = int(round((wi + 1) * Tfrm / valid_w))
                e = max(e, s + 1)
                seg = hidden[b, s:e, :]  # (segT, H)
                w_emb = seg.mean(dim=0)  # (H,)
                word_pred[b, wi] = self.word_head(w_emb).squeeze(-1)

        return {
            "sent_pred": sent_pred,
            "prosody_feat_pred": prosody_feat_pred,
            "word_pred": word_pred,
        }


# -------------------------
# Loss + Eval
# -------------------------
def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    w_sent: float,
    w_pfeat: float,
    w_words: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = outputs["sent_pred"].device
    sent_targets = batch["sent_targets"].to(device)         # (B,5)
    prosody_feats = batch["prosody_feats"].to(device)       # (B,K)
    word_scores = batch["word_scores"].to(device)           # (B,W)
    word_mask = batch["word_mask"].to(device)               # (B,W)

    sent_loss = F.smooth_l1_loss(outputs["sent_pred"], sent_targets)
    pfeat_loss = F.smooth_l1_loss(outputs["prosody_feat_pred"], prosody_feats)

    if word_mask.any():
        wp = outputs["word_pred"][word_mask]
        wt = word_scores[word_mask]
        word_loss = F.smooth_l1_loss(wp, wt)
    else:
        word_loss = torch.tensor(0.0, device=device)

    total = w_sent * sent_loss + w_pfeat * pfeat_loss + w_words * word_loss
    logs = {
        "loss_total": float(total.detach().cpu()),
        "loss_sent": float(sent_loss.detach().cpu()),
        "loss_prosody_feat": float(pfeat_loss.detach().cpu()),
        "loss_words": float(word_loss.detach().cpu()),
    }
    return total, logs


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, accelerator: Any) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_steps = 0

    for batch in loader:
        batch = move_batch_to_device(batch, accelerator.device)
        outputs = model(
            input_values=batch["input_values"],
            attention_mask=batch["attention_mask"],
            word_mask=batch["word_mask"],
        )
        loss, _ = compute_loss(outputs, batch, 1.0, 1.0, 1.0)
        loss = accelerator.gather(loss.detach()).mean()
        total_loss += float(loss.cpu())
        n_steps += 1

    return {"val_loss": total_loss / max(1, n_steps)}


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="mispeech/speechocean762", help="HF dataset name/path")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--valid_split", type=str, default="test")
    ap.add_argument("--model", type=str, default="facebook/hubert-base-ls960")

    ap.add_argument("--out_dir", type=str, default="ckpt_hubert_multitask")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--freeze_fe", action="store_true")

    ap.add_argument("--max_words", type=int, default=60)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--w_sent", type=float, default=1.0)
    ap.add_argument("--w_pfeat", type=float, default=0.5)
    ap.add_argument("--w_words", type=float, default=1.0)

    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)
    accelerator = create_accelerator()

    # Load HF dataset
    ds = load_dataset(args.dataset)

    # Keep raw audio payload (bytes/path), decode in collator.
    ds = ds.cast_column("audio", Audio(decode=False))

    train_ds = ds[args.train_split]
    valid_ds = ds[args.valid_split] if args.valid_split in ds else None

    # Fail fast on schema issues with explicit row references.
    preflight_n = min(32, len(train_ds))
    for i in range(preflight_n):
        validate_example_schema(train_ds[i], args.train_split, i)
    if valid_ds is not None:
        preflight_n_val = min(32, len(valid_ds))
        for i in range(preflight_n_val):
            validate_example_schema(valid_ds[i], args.valid_split, i)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    collator = Collator(
        feature_extractor=feature_extractor,
        sample_rate=16000,
        max_words=args.max_words,
        split_name=args.train_split,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=Collator(
                feature_extractor=feature_extractor,
                sample_rate=16000,
                max_words=args.max_words,
                split_name=args.valid_split,
            ),
        )

    model = HubertMultiTask(
        model_name=args.model,
        prosody_feat_dim=4,
        dropout=args.dropout,
        freeze_fe=args.freeze_fe,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, accelerator.device)
            outputs = model(
                input_values=batch["input_values"],
                attention_mask=batch["attention_mask"],
                word_mask=batch["word_mask"],
            )
            loss, logs = compute_loss(outputs, batch, args.w_sent, args.w_pfeat, args.w_words)
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running += logs["loss_total"]
            if accelerator.is_main_process and step % args.log_every == 0:
                print(f"[epoch {epoch} step {step}] train_loss={running / args.log_every:.4f}")
                running = 0.0

        if valid_loader is not None:
            metrics = eval_epoch(model, valid_loader, accelerator)
            if accelerator.is_main_process:
                print(f"== epoch {epoch} ==")
                print(metrics)

                if metrics["val_loss"] < best_val:
                    best_val = metrics["val_loss"]
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save(unwrapped.state_dict(), os.path.join(args.out_dir, "best.pt"))
                    print(f"Saved best to {os.path.join(args.out_dir, 'best.pt')}")
        else:
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), os.path.join(args.out_dir, f"epoch_{epoch}.pt"))
                print(f"Saved {os.path.join(args.out_dir, f'epoch_{epoch}.pt')}")

if __name__ == "__main__":
    main()
