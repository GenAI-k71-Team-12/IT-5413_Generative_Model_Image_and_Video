"""Baseline U-Net with GroupNorm (used for Full L1, Masked L1, Branch B, and Branch C)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Block(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """4-channel input (RGB + mask) U-Net returning harmonized RGB in [0, 1]."""

    def __init__(self, b: int = 32):
        super().__init__()
        self.enc = nn.ModuleList(
            [Block(4, b), Block(b, b * 2), Block(b * 2, b * 4), Block(b * 4, b * 8), Block(b * 8, b * 16)]
        )
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(b * 16, b * 8, 2, 2),
                nn.ConvTranspose2d(b * 8, b * 4, 2, 2),
                nn.ConvTranspose2d(b * 4, b * 2, 2, 2),
                nn.ConvTranspose2d(b * 2, b, 2, 2),
            ]
        )
        self.dec = nn.ModuleList(
            [Block(b * 16, b * 8), Block(b * 8, b * 4), Block(b * 4, b * 2), Block(b * 2, b)]
        )
        self.head = nn.Conv2d(b, 3, 1)

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < 4:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, block, skip in zip(self.up, self.dec, reversed(skips)):
            x = block(torch.cat([up(x), skip], dim=1))
        return self.head(x).sigmoid()
