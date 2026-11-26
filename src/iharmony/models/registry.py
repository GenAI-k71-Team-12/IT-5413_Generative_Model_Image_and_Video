"""Model registry, checkpoint loader, and unified inference wrapper for all 8+1 models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from iharmony.models.ht_plus import HTGenerator
from iharmony.models.improved_unet_v2 import ImprovedHarmonizationUNet
from iharmony.models.improved_unet_v3 import ImprovedHarmonizationUNetV3
from iharmony.models.rainnet import RAIN, RainNet
from iharmony.models.unet import UNet
from iharmony.utils import resolve_run_file, strip_module_prefix


def build_default_models_spec(
    candidate_runs_dirs: list[Path],
    include_masked_l1_30ep: bool = False,
) -> dict[str, dict]:
    """Return ordered registry covering Input Composite + all 8 models in runs/."""
    models: dict[str, dict] = {
        "0. Input Composite (No Harm)": {
            "slug": "00_input_composite",
            "short": "Input\nComposite",
            "arch": "identity",
            "path": None,
            "batch_size": 32,
            "amp": False,
        },
        "1. Baseline U-Net (Full L1)": {
            "slug": "01_unet_full_l1",
            "short": "U-Net\nFull L1",
            "arch": "unet",
            "path": resolve_run_file("unet_full_l1_all4_bs32_seed42/U-NET_Full_L1_checkpoint.pt", candidate_runs_dirs),
            "batch_size": 32,
            "amp": True,
        },
        "2. Baseline U-Net (Masked L1)": {
            "slug": "02_unet_masked_l1",
            "short": "U-Net\nMasked L1",
            "arch": "unet",
            "path": resolve_run_file("unet_masked_l1_50ep_all4_bs32_seed42/best.pt", candidate_runs_dirs),
            "batch_size": 32,
            "amp": True,
        },
        "3. U-Net + 1 PatchGAN (Branch B)": {
            "slug": "03_unet_branchB_1patchgan",
            "short": "U-Net +\n1-PatchGAN",
            "arch": "unet",
            "path": resolve_run_file("unet_branchB_1patchgan_50ep_all4_bs32_seed42/best.pt", candidate_runs_dirs),
            "batch_size": 32,
            "amp": True,
        },
        "4. U-Net + 3 PatchGAN (Branch C)": {
            "slug": "04_unet_branchC_3patchgan",
            "short": "U-Net +\n3-PatchGAN",
            "arch": "unet",
            "path": resolve_run_file("unet_branchC_3patchgan_50ep_all4_bs32_seed42/best.pt", candidate_runs_dirs),
            "batch_size": 32,
            "amp": True,
        },
        "5. Improved U-Net v2 (Hybrid)": {
            "slug": "05_improved_unet_v2_hybrid",
            "short": "Improved\nU-Net v2",
            "arch": "improved_unet_v2",
            "path": resolve_run_file("improved_unet_v2_hybrid_probe10ep_bs32_seed42/best.pt", candidate_runs_dirs),
            "batch_size": 32,
            "amp": True,
        },
        "6. Improved U-Net v3 (DINOv2 + Lab)": {
            "slug": "06_improved_unet_v3_dinov2_lab",
            "short": "U-Net v3\nDINOv2+Lab",
            "arch": "improved_unet_v3",
            "path": resolve_run_file("improved_unet_v3_dinov2_lab_30ep_bs32_seed42/best.pt", candidate_runs_dirs),
            "batch_size": 32,
            "amp": True,
        },
        "7. RainNet Pretrained (CVPR 2021)": {
            "slug": "07_rainnet_pretrained",
            "short": "RainNet\nPretrained",
            "arch": "rainnet",
            "path": resolve_run_file("RainNet_Pretrained/RainNet_net_G_last.pth", candidate_runs_dirs),
            "batch_size": 32,
            "amp": False,
        },
        "8. HT+ Pretrained (HT_2H9L)": {
            "slug": "08_ht_plus_pretrained",
            "short": "HT+ 2H9L\nPretrained",
            "arch": "ht_plus",
            "path": resolve_run_file("ht_plus_pretrained/HT_2H9L_allihd_latest_net_G.pth", candidate_runs_dirs),
            "batch_size": 16,
            "amp": False,
        },
    }
    if include_masked_l1_30ep:
        models["2b. U-Net Masked L1 (30ep Milestone)"] = {
            "slug": "02b_unet_masked_l1_30ep",
            "short": "Masked L1\n(30ep)",
            "arch": "unet",
            "path": resolve_run_file("unet_masked_l1_50ep_all4_bs32_seed42/best_30ep.pt", candidate_runs_dirs),
            "batch_size": 32,
            "amp": True,
        }
    return models


def load_harmonization_model(
    arch: str,
    ck_path: Optional[Path],
    device: torch.device,
    expected_fp: Optional[str] = None,
) -> tuple[Optional[torch.nn.Module], dict]:
    """Load any supported harmonization architecture with strict weight verification."""
    if arch == "identity":
        return None, {"epoch": "-", "trainable_params_M": 0.0, "total_params_M": 0.0, "val_fMSE": None}

    if ck_path is None:
        raise ValueError(f"Checkpoint path is required for architecture '{arch}'")

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)

    if arch in ("unet", "improved_unet_v2", "improved_unet_v3"):
        if expected_fp is not None and "fingerprint" in ck and ck["fingerprint"] != expected_fp:
            raise ValueError(f"Split fingerprint mismatch for {ck_path.name}!")
        cfg = ck.get("cfg", {})
        base_ch = cfg.get("base_channels", 32)
        num_heads = cfg.get("num_heads", 8)
        if arch == "unet":
            model = UNet(base_ch)
        elif arch == "improved_unet_v2":
            model = ImprovedHarmonizationUNet(base_channels=base_ch, num_heads=num_heads)
        else:
            model = ImprovedHarmonizationUNetV3(base_channels=base_ch, num_heads=num_heads)
        model.load_state_dict(strip_module_prefix(ck["model"]), strict=True)
        val_fmse = ck.get("val", {}).get("all", {}).get("fMSE")
        meta = {
            "epoch": ck.get("epoch", "?"),
            "val_fMSE": round(float(val_fmse), 2) if val_fmse is not None else None,
        }
    elif arch == "rainnet":
        model = RainNet(input_nc=3, output_nc=3, ngf=64, norm_layer=RAIN, use_dropout=True, use_attention=True)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(strip_module_prefix(sd), strict=True)
        meta = {"epoch": "Pre", "val_fMSE": None}
    elif arch == "ht_plus":
        model = HTGenerator(input_nc=3, output_nc=3, ksize=4, dim=256, nhead=2, num_layers=9, dim_forward=2, size=256)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(strip_module_prefix(sd), strict=True)
        meta = {"epoch": "Pre", "val_fMSE": None}
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model = model.to(device).eval()
    meta["total_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    meta["trainable_params_M"] = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)
    return model, meta


@torch.inference_mode()
def infer_raw_and_out(
    model: Optional[torch.nn.Module],
    arch: str,
    c: Tensor,
    m: Tensor,
    use_amp: bool = True,
) -> tuple[Tensor, Tensor]:
    """Run unified inference returning (raw, blended_out) as float32 tensors in [0, 1]."""
    if arch == "identity" or model is None:
        return c, c

    if arch == "rainnet":
        raw_m11 = model(c * 2.0 - 1.0, m).float()
        raw = ((raw_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
    elif arch == "ht_plus":
        raw = model(torch.cat([c, m], dim=1)).float().clamp(0.0, 1.0)
    else:
        with torch.autocast("cuda", dtype=torch.float16, enabled=(use_amp and c.is_cuda)):
            raw = model(torch.cat([c, m], dim=1))
        raw = raw.float().clamp(0.0, 1.0)

    out = (raw * m + c * (1.0 - m)).clamp(0.0, 1.0)
    return raw, out
