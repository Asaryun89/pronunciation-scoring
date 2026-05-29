"""
Self-supervised pre-training entry point.

Two usage modes
───────────────
1. Prepare k-means targets (run once before training):

       python pretrain.py --config configs/pretrain.yaml --prepare-kmeans

   This extracts MFCC features (iter=1) or model features (iter=2),
   fits k-means, and saves labels to the path specified in the config.
   Training does NOT start; run again without the flag to train.

2. Train:

       python pretrain.py --config configs/pretrain.yaml
       python pretrain.py --config configs/pretrain.yaml --resume checkpoints/pretrain/step_0010000.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data.pretrain_dataset import PretrainDataset, pretrain_collate_fn
from model import MultiResHuBERT
from training.kmeans import KMeansQuantizer
from training.logger import TrainLogger
from training.local_logger import LocalRunLogger
from training.pretrain_trainer import PretrainConfig, PretrainTrainer
from training.checkpoint_syncer import build_syncer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    """
    Load a YAML config and resolve every relative path value so it is
    absolute, anchored to the directory that contains the config file.

    This means you can run the scripts from *any* working directory and
    the paths will still resolve correctly.
    """
    config_file = Path(path).resolve()
    config_dir  = config_file.parent

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Keys whose string values are file-system paths that need anchoring.
    _PATH_KEYS = {
        ("data",     "labels_path"),
        ("data",     "kmeans_path"),
        ("kmeans",   "model_checkpoint"),
        ("training", "output_dir"),
        ("logging",  "tb_log_dir"),
    }

    for section, key in _PATH_KEYS:
        if section not in cfg:
            continue
        raw = cfg[section].get(key)
        if raw and raw != "null" and not Path(raw).is_absolute():
            cfg[section][key] = str(config_dir / raw)

    # Top-level pretrained_checkpoint (finetune config)
    if "pretrained_checkpoint" in cfg and cfg["pretrained_checkpoint"]:
        raw = cfg["pretrained_checkpoint"]
        if not Path(raw).is_absolute():
            cfg["pretrained_checkpoint"] = str(config_dir / raw)

    return cfg



def _load_hf_dataset(cfg: dict):
    """Load and return the training split from the HuggingFace Hub, cast to 16 kHz."""
    try:
        from datasets import load_dataset, Audio
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required.\n"
            "Install it with:  pip install datasets"
        ) from exc

    data_cfg   = cfg["data"]
    hf_id      = data_cfg.get("hf_dataset_id", "mispeech/speechocean762")
    split      = data_cfg.get("train_split", "train")
    cache_dir  = data_cfg.get("hf_cache_dir", None)

    log.info("Loading HF dataset: %s  split=%s", hf_id, split)
    hf_ds = load_dataset(hf_id, split=split, cache_dir=cache_dir)
    hf_ds = hf_ds.cast_column("audio", Audio(sampling_rate=16_000))
    log.info("  %d utterances loaded.", len(hf_ds))
    return hf_ds


def prepare_kmeans(cfg: dict) -> None:
    """
    Fit k-means on the HF dataset and write cluster labels to disk.
    Does NOT start training.
    """
    km_cfg    = cfg["kmeans"]
    data_cfg  = cfg["data"]
    iteration = km_cfg.get("iteration", 1)

    log.info("── K-means preparation  (iteration %d) ──", iteration)

    hf_ds = _load_hf_dataset(cfg)

    q = KMeansQuantizer(
        n_clusters_hi = km_cfg["n_clusters_hi"],
        n_clusters_lo = km_cfg.get("n_clusters_lo", None),
    )

    if iteration == 1:
        log.info("Extracting MFCC features …")
        q.fit_from_mfcc(hf_ds, n_mfcc=km_cfg.get("n_mfcc", 39))
    else:
        ckpt_path = km_cfg.get("model_checkpoint")
        if not ckpt_path:
            raise ValueError("kmeans.model_checkpoint must be set for iteration 2.")
        log.info("Loading model checkpoint: %s", ckpt_path)
        model = MultiResHuBERT(**cfg["model"])
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        log.info("Extracting model (f1) features …")
        q.fit_from_model(
            model,
            hf_ds,
            device=cfg["training"].get("device", "cpu"),
        )

    log.info("Assigning cluster labels …")
    q.assign_and_save(
        data_cfg["labels_path"],
        downsample_stride=cfg["model"].get("downsample_stride", 2),
    )

    kmeans_out = data_cfg.get("kmeans_path", "../data/kmeans/kmeans.pkl")
    q.save(kmeans_out)
    log.info("Done.  Run without --prepare-kmeans to start training.")


# ─────────────────────────────────────────────────────────────────────────────
# Build helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_loader(cfg: dict) -> torch.utils.data.DataLoader:
    data_cfg = cfg["data"]
    ds = PretrainDataset(
        labels_path    = data_cfg["labels_path"],
        hf_dataset_id  = data_cfg.get("hf_dataset_id", "mispeech/speechocean762"),
        split          = data_cfg.get("train_split", "train"),
        max_duration_s = data_cfg.get("max_duration_s", 20.0),
        cnn_stride     = cfg["kmeans"].get("cnn_stride", 320),
        cache_dir      = data_cfg.get("hf_cache_dir", None),
    )
    log.info("Pre-train dataset: %d utterances", len(ds))
    return torch.utils.data.DataLoader(
        ds,
        batch_size  = data_cfg["batch_size"],
        shuffle     = True,
        num_workers = data_cfg.get("num_workers", 4),
        collate_fn  = pretrain_collate_fn,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,
    )


def build_model(cfg: dict) -> MultiResHuBERT:
    model_cfg = dict(cfg["model"])
    # Force pretrain mode regardless of config value
    model_cfg["pretrain"] = True
    model = MultiResHuBERT(**model_cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info(
        "Model: %s | trainable %.1fM / %.1fM params",
        cfg["model"]["hubert_model_name"],
        n_train / 1e6, n_total / 1e6,
    )
    return model


def build_logger(cfg: dict, run_cfg: dict) -> TrainLogger:
    log_cfg = cfg.get("logging", {})
    return TrainLogger(
        use_wandb  = log_cfg.get("use_wandb", False),
        use_tb     = log_cfg.get("use_tb", False),
        tb_log_dir = log_cfg.get("tb_log_dir", "logs/pretrain"),
        project    = log_cfg.get("project", "multi-res-hubert"),
        run_name   = log_cfg.get("run_name", "pretrain"),
        config     = run_cfg,
    )


def build_local_logger(cfg: dict, resume_dir: str | None = None) -> LocalRunLogger:
    log_cfg  = cfg.get("logging", {})
    log_root = log_cfg.get("tb_log_dir", "logs/pretrain")
    return LocalRunLogger(
        log_root   = log_root,
        mode       = "pretrain",
        config     = cfg,
        resume_dir = resume_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-res HuBERT pre-training")
    parser.add_argument("--config",         default="configs/pretrain.yaml")
    parser.add_argument("--prepare-kmeans", action="store_true",
                        help="Fit k-means and save labels, then exit.")
    parser.add_argument("--resume",         default=None,
                        help="Path to a checkpoint to resume from.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # ── K-means preparation mode ──────────────────────────────────────────
    if args.prepare_kmeans:
        prepare_kmeans(cfg)
        return

    # ── Check labels exist ────────────────────────────────────────────────
    labels_path = cfg["data"]["labels_path"]
    if not Path(labels_path).exists():
        log.error(
            "Labels file not found: %s\n"
            "Run:  python pretrain.py --config %s --prepare-kmeans",
            labels_path, args.config,
        )
        sys.exit(1)

    # ── Determine resume_dir for local logger ─────────────────────────────
    resume_run_dir = None
    if args.resume:
        try:
            ckpt_meta = torch.load(args.resume, map_location="cpu")
            resume_run_dir = ckpt_meta.get("run_dir", None)
        except Exception:
            pass

    # ── Build components ──────────────────────────────────────────────────
    train_loader = build_loader(cfg)
    model        = build_model(cfg)
    logger       = build_logger(cfg, cfg)
    local_logger = build_local_logger(cfg, resume_dir=resume_run_dir)
    syncer       = build_syncer(cfg)

    train_cfg = cfg["training"]
    pretrain_config = PretrainConfig(
        output_dir      = train_cfg.get("output_dir",    "checkpoints/pretrain"),
        epochs          = train_cfg.get("epochs",        100),
        learning_rate   = train_cfg.get("learning_rate", 1e-4),
        weight_decay    = train_cfg.get("weight_decay",  1e-2),
        warmup_steps    = train_cfg.get("warmup_steps",  10_000),
        grad_clip       = train_cfg.get("grad_clip",     1.0),
        use_amp         = train_cfg.get("use_amp",       True),
        log_every                = train_cfg.get("log_every",               100),
        save_best_only           = train_cfg.get("save_best_only",          True),
        save_every               = train_cfg.get("save_every",             5_000),
        keep_last_n_checkpoints  = train_cfg.get("keep_last_n_checkpoints",    3),
        device                   = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        hi_loss_weight  = train_cfg.get("hi_loss_weight", 1.0),
        lo_loss_weight  = train_cfg.get("lo_loss_weight", 1.0),
    )

    trainer = PretrainTrainer(model, train_loader, pretrain_config, logger, local_logger, syncer)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
