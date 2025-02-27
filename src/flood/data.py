"""Flood-risk dataset: (DEM, rainfall, soil) -> depth."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FloodPatchDataset(Dataset):
    def __init__(self, features: np.ndarray, target: np.ndarray):
        if features.ndim != 4:
            raise ValueError("features must be (N, C, H, W)")
        if target.ndim != 3:
            raise ValueError("target must be (N, H, W)")
        if features.shape[0] != target.shape[0]:
            raise ValueError("batch size mismatch")
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.from_numpy(self.features[i]),
                torch.from_numpy(self.target[i]).unsqueeze(0))


def synthesize(n_samples: int = 16, size: int = 32,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dem = rng.uniform(0, 500, size=(n_samples, size, size)).astype(np.float32)
    rain = rng.exponential(20.0, size=(n_samples, size, size)).astype(np.float32)
    soil = rng.beta(2, 5, size=(n_samples, size, size)).astype(np.float32)
    features = np.stack([dem, rain, soil], axis=1)
    # synthetic depth target — low elevations + high rainfall + low permeability
    target = (rain / (dem + 1.0)) * (1.0 - soil)
    return features, target.astype(np.float32)
