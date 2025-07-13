import os
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"

from flood.train import TrainConfig, train_synthetic


def test_smoke_train_writes_checkpoint(tmp_path: Path):
    cfg = TrainConfig(in_channels=3, base_channels=4, n_levels=2,
                      epochs=2, batch_size=4, device="cpu",
                      checkpoint_dir=str(tmp_path / "ckpt"),
                      mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}")
    out = train_synthetic(cfg)
    assert "best_val_loss" in out
    assert (tmp_path / "ckpt" / "best.pt").exists()
