"""Small U-Net for flood-risk regression on (DEM, rainfall, soil) rasters."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    in_channels: int = 3
    base_channels: int = 32
    n_levels: int = 4
    out_channels: int = 1


def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.GroupNorm(min(8, out_c), out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.GroupNorm(min(8, out_c), out_c),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig = UNetConfig()):
        super().__init__()
        self.cfg = cfg
        chs = [cfg.base_channels * (2 ** i) for i in range(cfg.n_levels)]

        self.encs = nn.ModuleList()
        prev = cfg.in_channels
        for c in chs:
            self.encs.append(_conv_block(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _conv_block(chs[-1], chs[-1] * 2)

        self.ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        prev = chs[-1] * 2
        for c in reversed(chs):
            self.ups.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.dec_convs.append(_conv_block(prev, c))
            prev = c

        self.head = nn.Conv2d(chs[0], cfg.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for enc in self.encs:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.dec_convs,
                                 reversed(skips), strict=True):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.head(x)
