"""
Run before resuming training to confirm all adaptations are applied correctly.

Usage (from the project root or fusion_c_train/):
    python verify_adaptations.py
    python verify_adaptations.py --config fusion_c_train/configs/finetune_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml


def verify(config_path: str = "configs/finetune_config.yaml") -> None:
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        print(f"[FAIL]  Config not found: {cfg_file.resolve()}")
        raise SystemExit(1)

    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))

    checks: dict[str, bool] = {
        "batch_size=32":             cfg["data"]["batch_size"] == 32,
        "lr_text_encoder=1e-5":      cfg["training"]["lr_text_encoder"] == 1e-5,
        "min_lr=1e-5":               cfg["training"]["min_lr"] == 1e-5,
        "warmup_steps=1000":         cfg["training"]["warmup_steps"] == 1000,
        "grad_accum=2":              cfg["training"].get("grad_accumulation_steps") == 2,
        "label_smooth=0.05":         cfg["training"].get("label_smooth_alpha") == 0.05,
        "text_encoder_dim=1024":     cfg["model"]["text_encoder_dim"] == 1024,
        "text_encoder=Qwen3":        "Qwen3" in cfg["model"]["text_encoder_name"],
    }

    all_passed = True
    print(f"\nVerifying: {cfg_file.resolve()}\n")
    for name, result in checks.items():
        status = "[ OK ]" if result else "[FAIL]"
        print(f"  {status}  {name}")
        if not result:
            all_passed = False

    # Runtime check: fusion head accepts 1024-dim text input.
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from model.fusion_head import FusionScoringHead

        speech_dim = cfg["model"]["speech_rep_dim"]
        text_dim   = cfg["model"]["text_encoder_dim"]
        head = FusionScoringHead(
            speech_dim = speech_dim,
            text_dim   = text_dim,
            hidden     = cfg["model"]["fusion_hidden"],
            n_scores   = cfg["model"]["n_scores"],
            dropout    = cfg["model"]["dropout"],
        )
        dummy_s = torch.randn(2, speech_dim)
        dummy_t = torch.randn(2, text_dim)
        out     = head(dummy_s, dummy_t)
        shape_ok = out.shape == (2, 4)
        status = "[ OK ]" if shape_ok else "[FAIL]"
        print(
            f"  {status}  fusion head ({speech_dim}+{text_dim}={speech_dim+text_dim}) "
            f"-> {tuple(out.shape)}"
        )
        if not shape_ok:
            all_passed = False
    except Exception as exc:
        print(f"  [FAIL]  fusion head runtime check failed: {exc}")
        all_passed = False

    if all_passed:
        print("\n[ OK ] All adaptations verified — safe to resume training\n")
    else:
        print("\n[FAIL] Fix failing checks before resuming\n")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify training adaptations")
    parser.add_argument(
        "--config",
        default="configs/finetune_config.yaml",
        help="Path to finetune_config.yaml (default: configs/finetune_config.yaml)",
    )
    args = parser.parse_args()
    verify(args.config)
