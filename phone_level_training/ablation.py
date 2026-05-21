"""
Ablation study: which features matter for phoneme-accuracy prediction?

Feature groups
--------------
  ssl   : 1024-dim HuBERT last-hidden-state, mean-pooled per phone
  gop   : 1-dim Goodness-of-Pronunciation (log P_target - max log P_other)
  dur   : 1-dim log-duration (frame count)
  embed : 64-dim learnable phoneme-identity embedding

Configurations tested
---------------------
  all          — all four groups (full model)
  ssl_only     — only SSL (rich acoustic, no explicit pronunciation signal)
  gop_only     — only GOP (classic pronunciation measure, no deep features)
  no_ssl       — GOP + dur + embed  (traditional features, no SSL)
  no_gop       — SSL + dur + embed  (ablate explicit pronunciation measure)
  no_dur       — SSL + GOP + embed  (ablate timing)
  no_embed     — SSL + GOP + dur    (ablate phone identity)

Usage
-----
  python -m phone_level_training.ablation
  python -m phone_level_training.ablation --configs all no_ssl gop_only

Results are printed as a table and saved to outputs/ablation_results.json.
"""

import argparse
import json
import os

from .config import PATHS
from train import train

CONFIGS = {
    #                         ssl    gop    dur    embed
    "all":        dict(use_ssl=True,  use_gop=True,  use_dur=True,  use_phone_embed=True),
    "ssl_only":   dict(use_ssl=True,  use_gop=False, use_dur=False, use_phone_embed=False),
    "gop_only":   dict(use_ssl=False, use_gop=True,  use_dur=False, use_phone_embed=False),
    "no_ssl":     dict(use_ssl=False, use_gop=True,  use_dur=True,  use_phone_embed=True),
    "no_gop":     dict(use_ssl=True,  use_gop=False, use_dur=True,  use_phone_embed=True),
    "no_dur":     dict(use_ssl=True,  use_gop=True,  use_dur=False, use_phone_embed=True),
    "no_embed":   dict(use_ssl=True,  use_gop=True,  use_dur=True,  use_phone_embed=False),
}


def run_ablation(selected: list[str]) -> dict:
    results = {}
    for name in selected:
        if name not in CONFIGS:
            raise ValueError(f"Unknown config '{name}'. Choose from: {list(CONFIGS)}")
        print(f"\n{'='*60}")
        print(f"  Config: {name}")
        print(f"  Flags : {CONFIGS[name]}")
        print(f"{'='*60}")
        metrics = train(**CONFIGS[name], tag=name)
        results[name] = metrics
        print(f"  Best  : mse={metrics['mse']:.4f}  mae={metrics['mae']:.4f}  pearson={metrics['pearson']:.4f}")

    return results


def print_table(results: dict) -> None:
    header = f"{'Config':<14}  {'MSE':>7}  {'MAE':>7}  {'Pearson':>8}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, m in sorted(results.items(), key=lambda x: -x[1]["pearson"]):
        print(f"{name:<14}  {m['mse']:7.4f}  {m['mae']:7.4f}  {m['pearson']:8.4f}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Phoneme-accuracy ablation study")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIGS.keys()),
        help="Which configurations to run (default: all)",
    )
    args = parser.parse_args()

    results = run_ablation(args.configs)
    print_table(results)

    os.makedirs(PATHS["output_dir"], exist_ok=True)
    out_path = os.path.join(PATHS["output_dir"], "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
