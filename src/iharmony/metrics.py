"""GPU-accelerated harmonization evaluation metrics and validation helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as msssim_ssim
from torch import Tensor
from torchvision.utils import save_image
from tqdm.auto import tqdm

METRIC_KEYS = ("MSE", "fMSE", "PSNR", "fPSNR", "SSIM", "fSSIM", "fDeltaE76")

_SSIM_WINDOW_CACHE: dict[tuple[int, int, torch.device, torch.dtype], Tensor] = {}


def _get_gaussian_window(
    window_size: int, channels: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    key = (window_size, channels, device, dtype)
    if key not in _SSIM_WINDOW_CACHE:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g1d = torch.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
        g1d = (g1d / g1d.sum()).unsqueeze(1)
        g2d = (g1d @ g1d.t()).unsqueeze(0).unsqueeze(0)
        win = g2d.expand(channels, 1, window_size, window_size).contiguous().to(device=device, dtype=dtype)
        _SSIM_WINDOW_CACHE[key] = win
    return _SSIM_WINDOW_CACHE[key]


def compute_fssim(img1: Tensor, img2: Tensor, mask: Tensor, window_size: int = 11) -> Tensor:
    """Compute per-image foreground SSIM (fSSIM) following iHarmony4 / HarmonyTransformer."""
    channels = img1.shape[1]
    win = _get_gaussian_window(window_size, channels, img1.device, img1.dtype)
    pad = window_size // 2
    mu1 = F.conv2d(img1, win, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, win, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, win, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, win, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, win, padding=pad, groups=channels) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return (ssim_map * mask).sum(dim=(1, 2, 3)) / (channels * mask.sum(dim=(1, 2, 3)).clamp_min(1.0))


def rgb_to_lab_d65(rgb: Tensor) -> Tensor:
    """Convert sRGB tensor in [0, 1] of shape (B, 3, H, W) to CIE Lab (D65 illuminant)."""
    rgb = rgb.clamp(0.0, 1.0)
    lin = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).clamp_min(0.0).pow(2.4), rgb / 12.92)
    r, g, b = lin[:, 0], lin[:, 1], lin[:, 2]
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / 1.00000
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0

    def _f(t: Tensor) -> Tensor:
        return torch.where(t > eps, t.clamp_min(1e-12).pow(1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    l_ch = 116.0 * fy - 16.0
    a_ch = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)
    return torch.stack([l_ch, a_ch, b_ch], dim=1)


def compute_batch_metrics(out: Tensor, y: Tensor, m: Tensor) -> dict[str, Tensor]:
    """Compute all 7 per-image evaluation metrics on a batch of [0, 1] float32 tensors."""
    err = (out - y).square()
    mse = err.mean(dim=(1, 2, 3))
    fmse = (err * m).sum(dim=(1, 2, 3)) / (3.0 * m.sum(dim=(1, 2, 3)).clamp_min(1.0))

    lab_out = rgb_to_lab_d65(out)
    lab_y = rgb_to_lab_d65(y)
    de_map = torch.sqrt(((lab_out - lab_y) ** 2).sum(dim=1, keepdim=True).clamp_min(1e-12))
    f_delta_e = (de_map * m).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp_min(1.0)

    return {
        "MSE": mse * (255.0 ** 2),
        "fMSE": fmse * (255.0 ** 2),
        "PSNR": -10.0 * torch.log10(mse.clamp_min(1e-12)),
        "fPSNR": -10.0 * torch.log10(fmse.clamp_min(1e-12)),
        "SSIM": msssim_ssim(out, y, data_range=1.0, size_average=False),
        "fSSIM": compute_fssim(out, y, m, window_size=11),
        "fDeltaE76": f_delta_e,
    }


def aggregate_metrics_df(df: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate per-image metrics DataFrame into rounded mean statistics."""
    out: dict[str, float | int] = {"n": int(len(df))}
    for key in METRIC_KEYS:
        if key in df.columns:
            out[key] = round(float(df[key].mean()), 4)
    return out


@torch.inference_mode()
def evaluate_validation(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    save_to: Optional[Path] = None,
    include_ssim: bool = False,
) -> dict[str, dict[str, float]]:
    """Fast validation evaluator matching the original training notebooks."""
    model.eval()
    sums: dict[str, Counter] = defaultdict(Counter)
    saved = False

    for batch in tqdm(loader, desc="evaluate", leave=False):
        c, m, y, subsets = batch[0], batch[1], batch[2], batch[3]
        c = c.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            raw = model(torch.cat([c, m], dim=1))
        raw = raw.float().clamp(0.0, 1.0)
        out = (raw * m + c * (1.0 - m)).clamp(0.0, 1.0)

        err = (out - y).square()
        mse = err.mean(dim=(1, 2, 3))
        fmse = (err * m).sum(dim=(1, 2, 3)) / (3.0 * m.sum(dim=(1, 2, 3)).clamp_min(1.0))
        values = {
            "raw_L1": (raw - y).abs().mean(dim=(1, 2, 3)),
            "MSE": mse * (255.0 ** 2),
            "fMSE": fmse * (255.0 ** 2),
            "PSNR": -10.0 * torch.log10(mse.clamp_min(1e-12)),
            "fPSNR": -10.0 * torch.log10(fmse.clamp_min(1e-12)),
        }
        if include_ssim:
            values["SSIM"] = msssim_ssim(out, y, data_range=1.0, size_average=False)

        for i, subset in enumerate(subsets):
            for label in ("all", subset):
                sums[label]["n"] += 1
                for k, v in values.items():
                    sums[label][k] += float(v[i].item())

        if save_to is not None and not saved:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            panels = torch.stack([c, m.expand_as(c), raw, out, y], dim=1).flatten(0, 1)
            save_image(panels[:30], str(save_to), nrow=5)
            saved = True

    return {
        s: {k: (v if k == "n" else v / d["n"]) for k, v in d.items()}
        for s, d in sums.items()
    }
