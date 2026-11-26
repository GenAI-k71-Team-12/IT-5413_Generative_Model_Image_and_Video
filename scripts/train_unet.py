#!/usr/bin/env python3
"""Train Baseline U-Net with Full L1 or Masked L1 loss on iHarmony4 (256x256)."""

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
from iharmony.losses import full_l1_loss, masked_l1_loss
from iharmony.metrics import evaluate_validation
from iharmony.models import UNet
from iharmony.utils import atomic_save, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Baseline U-Net (Full L1 or Masked L1).")
    p.add_argument("--data-dir", type=Path, required=True, help="Path to resized iHarmony4 256x256 directory.")
    p.add_argument("--run-dir", type=Path, required=True, help="Output directory for checkpoints and logs.")
    p.add_argument("--loss", choices=("full_l1", "masked_l1"), default="masked_l1")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--milestone", type=int, default=30, help="Epoch milestone for saving best_30ep.pt.")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="Resume from run-dir/last.pt if present.")
    p.add_argument("--run-test", action="store_true", help="Evaluate best.pt on 7,404 test images after training.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, deterministic=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    loss_name = "raw_full_image_l1" if args.loss == "full_l1" else "masked_l1"
    cfg = {
        "seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_milestone": args.milestone,
        "size": args.size,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "base_channels": args.base_channels,
        "lr": args.lr,
        "eta_min": args.eta_min,
        "val_fraction": args.val_fraction,
        "loss": loss_name,
        "architecture": "unet_groupnorm_v1",
    }

    records, fingerprint, official_counts = build_records(
        args.data_dir, seed=args.seed, val_fraction=args.val_fraction
    )
    print(f"Train: {len(records['train'])} | Val: {len(records['val'])} | Test: {len(records['test'])}")

    model = UNet(args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best, best_milestone, history = 0, float("inf"), float("inf"), []
    last_ckpt = args.run_dir / "last.pt"
    if args.resume and last_ckpt.is_file():
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        if ck["cfg"] != cfg or ck["fingerprint"] != fingerprint:
            raise RuntimeError("Config or split fingerprint changed; specify a new --run-dir.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, history = ck["epoch"], ck["best"], ck["history"]
        best_milestone = ck.get("best_30", best)
        torch.set_rng_state(ck["torch_rng"])
        if device.type == "cuda" and "cuda_rng" in ck:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        print(f"Resumed from epoch {start_epoch} (best={best:.2f}, best_{args.milestone}={best_milestone:.2f})")

    env_meta = {
        "cfg": cfg,
        "fingerprint": fingerprint,
        "official_counts": official_counts,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    (args.run_dir / "config.json").write_text(json.dumps(env_meta, indent=2), encoding="utf-8")

    val_loader = make_dataloader(records, "val", args.batch_size, args.size, args.workers, args.seed)
    loss_key = "train_L1" if args.loss == "full_l1" else "train_fL1"

    for epoch in range(start_epoch, args.epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        tic = time.perf_counter()
        model.train()
        total, count = 0.0, 0

        train_loader = make_dataloader(records, "train", args.batch_size, args.size, args.workers, args.seed, epoch)
        bar = tqdm(train_loader, desc=f"{args.loss} | Ep {epoch + 1}/{args.epochs}")
        for c, m, y, _ in bar:
            c = c.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                raw = model(torch.cat([c, m], dim=1))
                loss = full_l1_loss(raw, y) if args.loss == "full_l1" else masked_l1_loss(raw, y, m)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected; check data/AMP settings.")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = c.shape[0]
            total += loss.item() * bs
            count += bs
            bar.set_postfix(loss=f"{total / count:.5f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        cur_ep = epoch + 1
        scores = evaluate_validation(model, val_loader, device, save_to=args.run_dir / f"val_{cur_ep:03d}.png")
        cur_fmse = scores["all"]["fMSE"]
        improved = cur_fmse < best
        best = min(best, cur_fmse)

        lr_used = optimizer.param_groups[0]["lr"]
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        row = {
            "epoch": cur_ep,
            loss_key: total / count,
            "val": scores,
            "lr": lr_used,
            "seconds": time.perf_counter() - tic,
            "peak_VRAM_GB": (torch.cuda.max_memory_allocated() / (2 ** 30)) if device.type == "cuda" else 0.0,
        }
        history.append(row)

        payload = {"model": model.state_dict(), "cfg": cfg, "epoch": cur_ep, "fingerprint": fingerprint, "val": scores}
        if cur_ep <= args.milestone and cur_fmse < best_milestone:
            best_milestone = cur_fmse
            atomic_save(payload, args.run_dir / f"best_{args.milestone}ep.pt")
        if improved:
            atomic_save(payload, args.run_dir / "best.pt")

        ck = {
            "epoch": cur_ep,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": cfg,
            "fingerprint": fingerprint,
            "best": best,
            "best_30": best_milestone,
            "history": history,
            "torch_rng": torch.get_rng_state(),
        }
        if device.type == "cuda":
            ck["cuda_rng"] = torch.cuda.get_rng_state_all()
        atomic_save(ck, last_ckpt)
        (args.run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False))

    if args.run_test and (args.run_dir / "best.pt").is_file():
        ck = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        test_loader = make_dataloader(records, "test", args.batch_size, args.size, args.workers, args.seed)
        test_scores = evaluate_validation(
            model, test_loader, device, save_to=args.run_dir / "test_preview.png", include_ssim=True
        )
        (args.run_dir / "test_metrics.json").write_text(json.dumps(test_scores, indent=2), encoding="utf-8")
        print(json.dumps(test_scores, indent=2))


if __name__ == "__main__":
    main()
