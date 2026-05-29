"""
Entry point for fine-tuning Multi-resolution HuBERT on Speechocean762.

Usage
─────
    cd multi_res_hubert
    python train.py --config configs/base.yaml

    # resume from a checkpoint
    python train.py --config configs/base.yaml --resume checkpoints/base/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Allow running from the package root without installing.
sys.path.insert(0, str(Path(__file__).parent))

from data import Speechocean762Dataset, collate_fn
from model import MultiResHuBERT
from training.trainer import Trainer, TrainerConfig
from training.logger import TrainLogger
from training.local_logger import LocalRunLogger
from training.checkpoint_syncer import build_syncer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """
    Load YAML config and anchor every relative path to the config file's
    directory so scripts work regardless of the current working directory.
    """
    config_file = Path(path).resolve()
    config_dir  = config_file.parent

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _PATH_KEYS = {
        ("data",     "train_dir"),
        ("data",     "val_dir"),
        ("data",     "scores_path"),
        ("training", "output_dir"),
        ("logging",  "tb_log_dir"),
    }
    for section, key in _PATH_KEYS:
        if section not in cfg:
            continue
        raw = cfg[section].get(key)
        if raw and not Path(raw).is_absolute():
            cfg[section][key] = str(config_dir / raw)

    if cfg.get("pretrained_checkpoint"):
        raw = cfg["pretrained_checkpoint"]
        if not Path(raw).is_absolute():
            cfg["pretrained_checkpoint"] = str(config_dir / raw)

    return cfg



def build_loader(
    cfg: dict,
    split: str,
    augment: bool,
) -> DataLoader:
    data_cfg = cfg["data"]
    ds = Speechocean762Dataset(
        split=split,
        hf_dataset_id=data_cfg.get("hf_dataset_id", "mispeech/speechocean762"),
        max_duration_s=data_cfg["max_duration_s"],
        augment=augment,
        cache_dir=data_cfg.get("hf_cache_dir", None),
    )
    logger.info("  split=%s  %d utterances", split, len(ds))
    return DataLoader(
        ds,
        batch_size=data_cfg["batch_size"],
        shuffle=augment,
        num_workers=data_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=augment,   # avoid single-sample batches that break Pearson
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MultiResHuBERT")
    parser.add_argument("--config",      default="configs/finetune.yaml")
    parser.add_argument("--resume",      default=None,
                        help="Fine-tune checkpoint to resume training from.")
    parser.add_argument("--pretrained",  default=None,
                        help="Pre-training checkpoint to warm-start the backbone.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logger.info("Building data loaders …")
    train_loader = build_loader(cfg, cfg["data"]["train_split"], augment=cfg["data"]["augment"])
    val_loader   = build_loader(cfg, cfg["data"]["val_split"],   augment=False)

    logger.info("Building model: %s", cfg["model"]["hubert_model_name"])
    model = MultiResHuBERT(**cfg["model"])
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    logger.info("  trainable: %.1fM / %.1fM params (%.1f%%)",
                n_trainable / 1e6, n_total / 1e6,
                100 * n_trainable / n_total)

    # Warm-start from a pre-training checkpoint (only backbone weights)
    pretrained_path = args.pretrained or cfg.get("pretrained_checkpoint")
    if pretrained_path:
        ckpt = torch.load(pretrained_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        logger.info(
            "Loaded pre-trained weights from %s  "
            "(missing=%d, unexpected=%d)",
            pretrained_path, len(missing), len(unexpected),
        )

    # ── Determine resume_dir for local logger ─────────────────────────────
    resume_run_dir = None
    if args.resume:
        try:
            ckpt_meta = torch.load(args.resume, map_location="cpu")
            resume_run_dir = ckpt_meta.get("run_dir", None)
        except Exception:
            pass

    # Build loggers
    log_cfg = cfg.get("logging", {})
    train_logger = TrainLogger(
        use_wandb  = log_cfg.get("use_wandb", False),
        use_tb     = log_cfg.get("use_tb",    False),
        tb_log_dir = log_cfg.get("tb_log_dir", "logs/finetune"),
        project    = log_cfg.get("project",   "multi-res-hubert"),
        run_name   = log_cfg.get("run_name",  "finetune"),
        config     = cfg,
    )
    local_logger = LocalRunLogger(
        log_root   = log_cfg.get("tb_log_dir", "logs/finetune"),
        mode       = "finetune",
        config     = cfg,
        resume_dir = resume_run_dir,
    )

    train_cfg = cfg["training"]
    trainer_cfg = TrainerConfig(
        output_dir    = train_cfg.get("output_dir",    "checkpoints/finetune"),
        epochs        = train_cfg.get("epochs",        30),
        learning_rate = train_cfg.get("learning_rate", 1e-4),
        weight_decay  = train_cfg.get("weight_decay",  1e-2),
        warmup_steps  = train_cfg.get("warmup_steps",  500),
        grad_clip     = train_cfg.get("grad_clip",     1.0),
        use_amp       = train_cfg.get("use_amp",       True),
        log_every     = train_cfg.get("log_every",     50),
        eval_every    = train_cfg.get("eval_every",    1),
        save_best     = train_cfg.get("save_best",     True),
        device        = train_cfg.get("device",        "cuda" if torch.cuda.is_available() else "cpu"),
        score_weights = train_cfg.get("score_weights", [1.0, 1.0, 1.0, 1.0, 2.0]),
        mse_weight    = train_cfg.get("mse_weight",    1.0),
        corr_weight   = train_cfg.get("corr_weight",   0.5),
    )

    syncer  = build_syncer(cfg)
    trainer = Trainer(model, train_loader, val_loader, trainer_cfg, train_logger, local_logger, syncer)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
