#!/usr/bin/env python3
"""Train Improved U-Net v2 (Hybrid) or Improved U-Net v3 (DINOv2 + Lab) on iHarmony4."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
from tqdm.auto import tqdm

from iharmony.data import build_records, make_dataloader
from iharmony.losses import background_l1_loss, masked_l1_loss, masked_lab_l1_loss
from iharmony.metrics import evaluate_validation
from iharmony.models import ImprovedHarmonizationUNet, ImprovedHarmonizationUNetV3
from iharmony.utils import atomic_save, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Improved U-Net v2 (Hybrid) or v3 (DINOv2 + Lab).")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--version", choices=("v2", "v3"), default="v3")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-bg", type=float, default=0.1)
    p.add_argument("--lambda-lab", type=float, default=0.15, help="Used for v3 CIE Lab loss.")
    p.add_argument("--fg-color-aug-prob", type=float, default=0.35, help="Used for v3 foreground color augmentation.")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, deterministic=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    is_v3 = args.version == "v3"
    cfg = {
        "model_name": "improved_unet_v3_dinov2_lab" if is_v3 else "improved_unet_v2_hybrid",
        "seed": args.seed,
        "epochs": args.epochs,
        "size": args.size,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "base_channels": args.base_channels,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "eta_min": args.eta_min,
        "weight_decay": args.weight_decay if is_v3 else 0.0,
        "lambda_bg": args.lambda_bg,
        "lambda_lab": args.lambda_lab if is_v3 else 0.0,
        "fg_color_aug_prob": args.fg_color_aug_prob if is_v3 else 0.0,
        "val_fraction": args.val_fraction,
    }

    records, fingerprint, _ = build_records(args.data_dir, seed=args.seed, val_fraction=args.val_fraction)
    model = (
        ImprovedHarmonizationUNetV3(base_channels=args.base_channels, num_heads=args.num_heads).to(device)
        if is_v3
        else ImprovedHarmonizationUNet(base_channels=args.base_channels, num_heads=args.num_heads).to(device)
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best, history = 0, float("inf"), []
    last_ckpt = args.run_dir / "last.pt"
    if args.resume and last_ckpt.is_file():
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        if ck["fingerprint"] != fingerprint:
            raise RuntimeError("Split fingerprint mismatch on resume!")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_epoch, best, history = ck["epoch"], ck["best"], ck["history"]
        print(f"Resumed {args.version} from epoch {start_epoch}/{args.epochs} (best fMSE={best:.2f})")

    total_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    train_m = round(sum(p.numel() for p in trainable_params) / 1e6, 2)
    (args.run_dir / "config.json").write_text(
        json.dumps(
            {
                "cfg": cfg,
                "fingerprint": fingerprint,
                "trainable_params_M": train_m,
                "total_params_M": total_m,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    val_loader = make_dataloader(records, "val", args.batch_size, args.size, args.workers, args.seed)

    for epoch in range(start_epoch, args.epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        tic = time.perf_counter()
        model.train()
        tot_loss, tot_fl1, tot_bg, tot_lab, count = 0.0, 0.0, 0.0, 0.0, 0

        train_loader = make_dataloader(
            records,
            "train",
            args.batch_size,
            args.size,
            args.workers,
            args.seed,
            epoch,
            fg_color_aug_prob=cfg["fg_color_aug_prob"],
        )
        bar = tqdm(train_loader, desc=f"Improved {args.version} | Ep {epoch + 1}/{args.epochs}")
        for c, m, y, _ in bar:
            c = c.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                raw = model(torch.cat([c, m], dim=1))
                fl1 = masked_l1_loss(raw, y, m)
                bg_l1 = background_l1_loss(raw, c, m)
                loss = fl1 + cfg["lambda_bg"] * bg_l1
                if is_v3 and cfg["lambda_lab"] > 0:
                    lab_l1 = masked_lab_l1_loss(raw, y, m)
                    loss = loss + cfg["lambda_lab"] * lab_l1
                else:
                    lab_l1 = torch.zeros((), device=device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = c.shape[0]
            tot_loss += float(loss.item()) * bs
            tot_fl1 += float(fl1.item()) * bs
            tot_bg += float(bg_l1.item()) * bs
            tot_lab += float(lab_l1.item()) * bs
            count += bs
            bar.set_postfix(loss=f"{tot_loss / count:.5f}", fL1=f"{tot_fl1 / count:.5f}")

        cur_ep = epoch + 1
        scores = evaluate_validation(model, val_loader, device, save_to=args.run_dir / f"val_{cur_ep:03d}.png")
        cur_fmse = scores["all"]["fMSE"]
        improved = cur_fmse < best
        best = min(best, cur_fmse)

        lr_used = optimizer.param_groups[0]["lr"]
        scheduler.step()

        row = {
            "model": cfg["model_name"],
            "epoch": cur_ep,
            "train_loss": tot_loss / count,
            "train_fL1": tot_fl1 / count,
            "train_bgL1": tot_bg / count,
            "val": scores,
            "lr": lr_used,
            "seconds": time.perf_counter() - tic,
            "peak_VRAM_GB": (torch.cuda.max_memory_allocated() / (2 ** 30)) if device.type == "cuda" else 0.0,
        }
        if is_v3:
            row["train_labL1"] = tot_lab / count
        history.append(row)

        payload = {"model": model.state_dict(), "cfg": cfg, "epoch": cur_ep, "fingerprint": fingerprint, "val": scores}
        if improved:
            atomic_save(payload, args.run_dir / "best.pt")

        atomic_save(
            {
                "epoch": cur_ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": cfg,
                "fingerprint": fingerprint,
                "best": best,
                "history": history,
            },
            last_ckpt,
        )
        (args.run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
