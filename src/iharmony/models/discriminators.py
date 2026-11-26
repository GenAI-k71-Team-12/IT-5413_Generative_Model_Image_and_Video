"""Spectral-Normalized PatchGAN discriminators for Branch B (1-scale) and Branch C (3-scale)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """70x70 Spectral-Normalized PatchGAN discriminator (Branch B)."""

    def __init__(self, input_nc: int = 4, ndf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiScalePatchGANDiscriminator(nn.Module):
    """Multi-scale Spectral-Normalized PatchGAN discriminator (Branch C, num_D=3)."""

    def __init__(self, input_nc: int = 4, ndf: int = 64, num_d: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PatchGANDiscriminator(input_nc=input_nc, ndf=ndf) for _ in range(num_d)]
        )
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs: list[Tensor] = []
        cur = x
        for idx, disc in enumerate(self.discriminators):
            outputs.append(disc(cur))
            if idx < len(self.discriminators) - 1:
                cur = self.downsample(cur)
        return outputs
