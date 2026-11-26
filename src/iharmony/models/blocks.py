"""Shared architectural building blocks for Improved U-Net v2 and v3."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class GatedConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(num_groups=math.gcd(8, out_ch), num_channels=out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        feat, gate = self.conv(x).chunk(2, dim=1)
        return self.act(self.norm(feat)) * torch.sigmoid(gate)


class GatedDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(GatedConv2d(in_ch, out_ch), GatedConv2d(out_ch, out_ch))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class StandardDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        num_groups = math.gcd(8, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BilinearUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.up(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch: int, x_ch: int):
        super().__init__()
        inter_ch = max(x_ch // 2, 16)
        self.W_g = nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        return x * self.psi(self.W_g(g) + self.W_x(x))
