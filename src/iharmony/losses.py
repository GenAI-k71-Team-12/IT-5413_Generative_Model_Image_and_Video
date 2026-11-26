"""Loss functions for U-Net baselines, PatchGAN branches, and Improved U-Net v2/v3."""

from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from iharmony.metrics import rgb_to_lab_d65

TensorOrList = Union[Tensor, Sequence[Tensor]]


def full_l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Standard full-image L1 loss used by Baseline U-Net (Full L1)."""
    return F.l1_loss(pred, target)


def masked_l1_loss(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Foreground-normalized L1 loss used by Masked L1, Branch B/C, and Improved U-Net."""
    diff = torch.abs(pred - target) * mask
    per_img = diff.sum(dim=(1, 2, 3)) / (3.0 * mask.sum(dim=(1, 2, 3)).clamp_min(eps))
    return per_img.mean()


def background_l1_loss(pred: Tensor, comp: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Background preservation L1 loss used in Improved U-Net v2/v3."""
    bg = 1.0 - mask
    diff = torch.abs(pred - comp) * bg
    per_img = diff.sum(dim=(1, 2, 3)) / (3.0 * bg.sum(dim=(1, 2, 3)).clamp_min(eps))
    return per_img.mean()


def masked_lab_l1_loss(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Foreground-normalized L1 loss in CIE Lab (D65) space (scaled by 1/100) for v3."""
    lab_pred = rgb_to_lab_d65(pred.float()) / 100.0
    lab_tgt = rgb_to_lab_d65(target.float()) / 100.0
    diff = torch.abs(lab_pred - lab_tgt) * mask.float()
    per_img = diff.sum(dim=(1, 2, 3)) / (3.0 * mask.float().sum(dim=(1, 2, 3)).clamp_min(eps))
    return per_img.mean()


def lsgan_discriminator_loss(
    pred_real: TensorOrList,
    pred_fake: TensorOrList,
    real_label: float = 0.9,
) -> tuple[Tensor, list[float]]:
    """LSGAN discriminator loss supporting single-scale (Branch B) and multi-scale (Branch C)."""
    reals = [pred_real] if isinstance(pred_real, Tensor) else list(pred_real)
    fakes = [pred_fake] if isinstance(pred_fake, Tensor) else list(pred_fake)
    scale_losses = []
    for pr, pf in zip(reals, fakes):
        loss_s = 0.5 * (
            F.mse_loss(pr, torch.full_like(pr, real_label))
            + F.mse_loss(pf, torch.zeros_like(pf))
        )
        scale_losses.append(loss_s)
    total = sum(scale_losses) / len(scale_losses)
    return total, [float(l.detach().item()) for l in scale_losses]


def lsgan_generator_loss(pred_fake: TensorOrList) -> Tensor:
    """LSGAN generator adversarial loss supporting single-scale and multi-scale discriminators."""
    fakes = [pred_fake] if isinstance(pred_fake, Tensor) else list(pred_fake)
    losses = [F.mse_loss(pf, torch.ones_like(pf)) for pf in fakes]
    return sum(losses) / len(losses)
