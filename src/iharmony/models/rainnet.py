"""Self-contained RainNet (Ling et al., CVPR 2021) matching official net_G_last.pth weights."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RAIN(nn.Module):
    """Region-aware Adaptive Instance Normalization."""

    def __init__(self, dims_in: int, eps: float = 1e-5):
        super().__init__()
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.eps = eps

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode="nearest")
        mean_back, std_back = self.get_foreground_mean_std(x * (1 - mask), 1 - mask)
        normalized = (x - mean_back) / std_back
        normalized_background = (
            normalized * (1 + self.background_gamma[None, :, None, None])
            + self.background_beta[None, :, None, None]
        ) * (1 - mask)

        mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask)
        normalized = (x - mean_fore) / std_fore * std_back + mean_back
        normalized_foreground = (
            normalized * (1 + self.foreground_gamma[None, :, None, None])
            + self.foreground_beta[None, :, None, None]
        ) * mask
        return normalized_foreground + normalized_background

    def get_foreground_mean_std(self, region: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        s = torch.sum(region, dim=[2, 3])
        num = torch.sum(mask, dim=[2, 3])
        mu = s / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask) * mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var + self.eps)


def _inst_norm(ch: int) -> nn.InstanceNorm2d:
    return nn.InstanceNorm2d(ch, affine=False, track_running_stats=False)


class UnetBlockCodec(nn.Module):
    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer=RAIN,
        use_dropout: bool = False,
        use_attention: bool = False,
        enc: int | bool = True,
        dec: int | bool = True,
    ):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_dropout = use_dropout
        self.use_attention = use_attention
        if input_nc is None:
            input_nc = outer_nc

        if outermost:
            self.down = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
            self.submodule = submodule
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
        elif innermost:
            self.up = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False),
                nn.ReLU(True),
                nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.upnorm = norm_layer(outer_nc) if dec else _inst_norm(outer_nc)
        else:
            self.down = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.downnorm = norm_layer(inner_nc) if enc else _inst_norm(inner_nc)
            self.submodule = submodule
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.upnorm = norm_layer(outer_nc) if dec else _inst_norm(outer_nc)
            if use_dropout:
                self.dropout = nn.Dropout(0.5)

        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(outer_nc + input_nc, outer_nc + input_nc, kernel_size=1),
                nn.Sigmoid(),
            )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        if self.outermost:
            return self.up(self.submodule(self.down(x), mask))
        if self.innermost:
            ret = self.up(x)
            ret = self.upnorm(ret, mask) if isinstance(self.upnorm, RAIN) else self.upnorm(ret)
            ret = torch.cat([x, ret], dim=1)
            return self.attention(ret) * ret if self.use_attention else ret

        ret = self.down(x)
        ret = self.downnorm(ret, mask) if isinstance(self.downnorm, RAIN) else self.downnorm(ret)
        ret = self.submodule(ret, mask)
        ret = self.up(ret)
        ret = self.upnorm(ret, mask) if isinstance(self.upnorm, RAIN) else self.upnorm(ret)
        if self.use_dropout:
            ret = self.dropout(ret)
        ret = torch.cat([x, ret], dim=1)
        return self.attention(ret) * ret if self.use_attention else ret


class RainNet(nn.Module):
    """RainNet generator expecting composite in [-1, 1] and binary mask in [0, 1]."""

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        norm_layer=RAIN,
        norm_type_indicator=(0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1),
        use_dropout: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_nc = input_nc
        self.use_attention = use_attention
        norm_type_list = [_inst_norm, norm_layer]

        self.model_layer0 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.model_layer1 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False))
        self.model_layer1norm = norm_type_list[norm_type_indicator[0]](ngf * 2)

        self.model_layer2 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False))
        self.model_layer2norm = norm_type_list[norm_type_indicator[1]](ngf * 4)

        self.model_layer3 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False))
        self.model_layer3norm = norm_type_list[norm_type_indicator[2]](ngf * 8)

        unet_block = UnetBlockCodec(
            ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer,
            innermost=True, enc=norm_type_indicator[6], dec=norm_type_indicator[7],
        )
        unet_block = UnetBlockCodec(
            ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer,
            use_dropout=use_dropout, enc=norm_type_indicator[5], dec=norm_type_indicator[8],
        )
        unet_block = UnetBlockCodec(
            ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer,
            use_dropout=use_dropout, enc=norm_type_indicator[4], dec=norm_type_indicator[9],
        )
        self.unet_block = UnetBlockCodec(
            ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer,
            use_dropout=use_dropout, enc=norm_type_indicator[3], dec=norm_type_indicator[10],
        )

        self.model_layer11 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, 2, 1, bias=False))
        self.model_layer11norm = norm_type_list[norm_type_indicator[11]](ngf * 4)
        if use_attention:
            self.model_layer11att = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, 1, 1), nn.Sigmoid())

        self.model_layer12 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1, bias=False))
        self.model_layer12norm = norm_type_list[norm_type_indicator[12]](ngf * 2)
        if use_attention:
            self.model_layer12att = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, 1, 1), nn.Sigmoid())

        self.model_layer13 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1, bias=False))
        self.model_layer13norm = norm_type_list[norm_type_indicator[13]](ngf)
        if use_attention:
            self.model_layer13att = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 2, 1, 1), nn.Sigmoid())

        self.model_out = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1), nn.Tanh())

    @staticmethod
    def _apply_norm(layer: nn.Module, x: Tensor, mask: Tensor) -> Tensor:
        return layer(x, mask) if isinstance(layer, RAIN) else layer(x)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x0 = self.model_layer0(x)
        x1 = self._apply_norm(self.model_layer1norm, self.model_layer1(x0), mask)
        x2 = self._apply_norm(self.model_layer2norm, self.model_layer2(x1), mask)
        x3 = self._apply_norm(self.model_layer3norm, self.model_layer3(x2), mask)

        ox3 = self.unet_block(x3, mask)
        ox2 = torch.cat([x2, self._apply_norm(self.model_layer11norm, self.model_layer11(ox3), mask)], dim=1)
        if self.use_attention:
            ox2 = self.model_layer11att(ox2) * ox2

        ox1 = torch.cat([x1, self._apply_norm(self.model_layer12norm, self.model_layer12(ox2), mask)], dim=1)
        if self.use_attention:
            ox1 = self.model_layer12att(ox1) * ox1

        ox0 = torch.cat([x0, self._apply_norm(self.model_layer13norm, self.model_layer13(ox1), mask)], dim=1)
        if self.use_attention:
            ox0 = self.model_layer13att(ox0) * ox0

        return self.model_out(ox0)
