"""Training loop."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flood.data import FloodPatchDataset, synthesize
from flood.unet import UNet, UNetConfig

log = logging.getLogger(__name__)

try:
    import wandb
    _WANDB = True
except ImportError:  # pragma: no cover
    _WANDB = False


@dataclass
class TrainConfig:
    in_channels: int = 3
    base_channels: int = 32
    n_levels: int = 4
    epochs: int = 10
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.2
    seed: int = 20260514
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "flood-risk"
    wandb_project: str = "flood-risk"


def train(features: np.ndarray, target: np.ndarray, cfg: TrainConfig
          ) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    n = features.shape[0]
    n_val = max(1, int(n * cfg.val_fraction))
    train_ds = FloodPatchDataset(features[:-n_val], target[:-n_val])
    val_ds = FloodPatchDataset(features[-n_val:], target[-n_val:])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = UNet(UNetConfig(
        in_channels=cfg.in_channels, base_channels=cfg.base_channels,
        n_levels=cfg.n_levels,
    )).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    run_name = f"flood-{int(time.time())}"
    wb = None
    if _WANDB and os.environ.get("WANDB_MODE") != "disabled":
        wb = wandb.init(project=cfg.wandb_project, name=run_name,
                        config=cfg.__dict__, reinit=True)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_path = Path(cfg.checkpoint_dir) / "best.pt"
    best_val = float("inf")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg.__dict__)
        for ep in range(cfg.epochs):
            model.train(True)
            tot = 0.0
            n_seen = 0
            for x, y in train_loader:
                x = x.to(device); y = y.to(device)
                pred = model(x)
                loss = F.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * x.size(0)
                n_seen += x.size(0)

            model.train(False)
            val_loss = 0.0
            n_val_seen = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device); y = y.to(device)
                    val_loss += F.smooth_l1_loss(model(x), y).item() * x.size(0)
                    n_val_seen += x.size(0)
            val_loss /= max(n_val_seen, 1)

            row = {"train_loss": tot / max(n_seen, 1), "val_loss": val_loss}
            mlflow.log_metrics(row, step=ep)
            if wb is not None:
                wb.log(row, step=ep)
            log.info("ep=%03d train=%.4f val=%.4f", ep,
                     row["train_loss"], val_loss)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(str(best_path), artifact_path="checkpoints")

        mlflow.pytorch.log_model(model, artifact_path="model")

    if wb is not None:
        wb.finish()
    return {"best_val_loss": float(best_val)}


def train_synthetic(cfg: TrainConfig | None = None) -> dict[str, float]:
    cfg = cfg or TrainConfig()
    f, t = synthesize(n_samples=16, size=32)
    return train(f, t, cfg)
