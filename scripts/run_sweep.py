#!/usr/bin/env python3
"""
Multi-seed campaign runner for reproducing the 5-seed training cohorts.

Canonical seeds used across the 40-run BN-equalized primary cohort:
  42, 123, 456, 789, 1024
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import PRIMARY_BN_MODELS, UNNORMALIZED_VGG_MODELS

CANONICAL_SEEDS = [42, 123, 456, 789, 1024]


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed CNN benchmark sweep")
    parser.add_argument(
        "--cohort",
        choices=["primary_bn", "unnormalized_vgg", "all"],
        default="primary_bn",
        help="Which model cohort to train (default: primary_bn)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit model list overriding --cohort",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=CANONICAL_SEEDS,
        help="Random seeds to run (default: 42 123 456 789 1024)",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out_root", type=str, default="runs")
    parser.add_argument("--preresized_dir", type=str, default="")
    args = parser.parse_args()

    if args.models:
        models = args.models
    elif args.cohort == "primary_bn":
        models = PRIMARY_BN_MODELS
    elif args.cohort == "unnormalized_vgg":
        models = UNNORMALIZED_VGG_MODELS
    else:
        models = PRIMARY_BN_MODELS + UNNORMALIZED_VGG_MODELS

    for model in models:
        for seed in args.seeds:
            out_dir = Path(args.out_root) / f"{model}_s{seed}"
            cmd = [
                sys.executable,
                str(ROOT / "scripts/train.py"),
                "--model",
                model,
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--ckpt_dir",
                str(out_dir),
                "--log_dir",
                str(out_dir / "tensorboard"),
            ]
            if args.preresized_dir:
                cmd.extend(["--preresized_dir", args.preresized_dir])
            print(f"[sweep] Running {model} (seed={seed}) -> {out_dir}")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
