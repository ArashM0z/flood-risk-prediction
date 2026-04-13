# Flood Risk Prediction

Small U-Net for flood-depth regression on (DEM, rainfall, soil) feature
rasters. Includes a synthetic 32x32 fixture so the full training loop runs
in seconds on a laptop CPU.

## What's in the box

- `UNet(UNetConfig)` — encoder/decoder with configurable depth and base
  width, `GroupNorm` for batch-size-1 robustness, bilinear upsample to
  recover the input resolution for non-power-of-two inputs.
- `FloodPatchDataset` — wraps `(N, C, H, W)` features and `(N, H, W)`
  targets; raises on size mismatch.
- `train(features, target, cfg)` — AdamW + smooth-L1, gradient clipping,
  MLflow + W&B dual logging, best-checkpoint persistence by val loss.
- `flood-train --config configs/default.yaml --demo` runs end-to-end on
  the synthetic fixture in well under a minute.

## Quickstart

```bash
pip install -e ".[dev]"
WANDB_MODE=disabled make demo
flood-train --config configs/default.yaml \
            --features data/features.npz \
            --target data/targets.npz
```

## Layout

```
src/flood/
├── unet.py    # Small U-Net with GroupNorm + bilinear upsample
├── data.py    # FloodPatchDataset + synthesize()
├── train.py   # AdamW + smooth-L1 + MLflow + W&B
└── cli.py     # flood-train entrypoint
configs/       # default.yaml
tests/         # forward shapes, dataset mismatches, smoke train
```
