"""CLI."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from flood.data import synthesize
from flood.train import TrainConfig, train


def train_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--features", type=Path,
                   help=".npz with key 'data' shape (N, C, H, W)")
    p.add_argument("--target", type=Path,
                   help=".npz with key 'data' shape (N, H, W)")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = yaml.safe_load(args.config.read_text()) or {}
    cfg = TrainConfig(**raw)
    if args.demo or args.features is None:
        f, t = synthesize()
    else:
        f = np.load(args.features)["data"]
        t = np.load(args.target)["data"]
    out = train(f, t, cfg)
    for k, v in out.items():
        print(f"{k}={v:.4f}")
    return 0
