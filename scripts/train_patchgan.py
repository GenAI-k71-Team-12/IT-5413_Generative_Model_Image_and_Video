#!/usr/bin/env python3
"""Fine-tune U-Net from epoch-30 milestone with 1-PatchGAN (Branch B) or 3-MultiScale PatchGAN (Branch C)."""

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
from iharmony.losses import lsgan_discriminator_loss, lsgan_generator_loss, masked_l1_loss
from iharmony.metrics import evaluate_validation
from iharmony.models import MultiScalePatchGANDiscriminator, PatchGANDiscriminator, UNet
from iharmony.utils import atomic_save, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net + SN-PatchGAN (Branch B: --num-d 1, Branch C: --num-d 3).")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--base-ckpt", type=Path, required=True, help="Path to best_30ep.pt from Masked L1 run.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--num-d", type=int, choices=(1, 3), default=1, help="1 = Branch B, 3 = Branch C multi-scale.")
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--start-epoch", type=int, default=30)
    p.add_argument("--total-epochs", type=int, default=50)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--lr-g", type=float, default=5e-5)
    p.add_argument("--lr-d", type=float, default=1e-4)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--lambda-l1", type=float, default=100.0)
    p.add_argument("--real-label", type=float, default=0.9)
    p.add_argument("--n-critic-warmup", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, deterministic=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    branch = "B" if args.num_d == 1 else "C"
    extra_epochs = args.total_epochs - args.start_epoch
    cfg = {
        "branch": branch,
        "seed": args.seed,
        "start_epoch": args.start_epoch,
        "total_epochs": args.total_epochs,
        "extra_epochs": extra_epochs,
        "size": args.size,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "base_channels": args.base_channels,
        "num_D": args.num_d,
        "ndf": args.ndf,
        "lr_G": args.lr_g,
        "lr_D": args.lr_d,
        "eta_min": args.eta_min,
        "lambda_l1": args.lambda_l1,
        "real_label": args.real_label,
        "n_critic_warmup_epochs": args.n_critic_warmup,
        "val_fraction": args.val_fraction,
        "loss": "masked_l1_plus_1_patchgan_balanced" if args.num_d == 1 else "masked_l1_plus_3_multiscale_patchgan",
        "architecture": (
            "unet_groupnorm_v1_sn_patchgan" if args.num_d == 1 else "unet_groupnorm_v1_sn_multiscale_patchgan"
        ),
    }

    records, fingerprint, _ = build_records(args.data_dir, seed=args.seed, val_fraction=args.val_fraction)
    model = UNet(args.base_channels).to(device)
    net_d = (
        PatchGANDiscriminator(input_nc=4, ndf=args.ndf).to(device)
        if args.num_d == 1
        else MultiScalePatchGANDiscriminator(input_nc=4, ndf=args.ndf, num_d=args.num_d).to(device)
    )

    opt_g = torch.optim.Adam(model.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=max(1, extra_epochs - 1), eta_min=args.eta_min)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=max(1, extra_epochs), eta_min=args.eta_min)
    scaler_g = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scaler_d = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_ep = args.start_epoch + 1
    best_overall, best_gan, history = float("inf"), float("inf"), []
    last_ckpt = args.run_dir / "last.pt"

    if args.resume and last_ckpt.is_file():
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        if ck["fingerprint"] != fingerprint:
            raise RuntimeError("Split fingerprint mismatch on resume!")
        model.load_state_dict(ck["model"])
        net_d.load_state_dict(ck["netD"])
        opt_g.load_state_dict(ck["opt_G"])
        opt_d.load_state_dict(ck["opt_D"])
        for opt in (opt_g, opt_d):
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        sched_g.load_state_dict(ck["sched_G"])
        sched_d.load_state_dict(ck["sched_D"])
        scaler_g.load_state_dict(ck["scaler_G"])
        scaler_d.load_state_dict(ck["scaler_D"])
        start_ep = ck["epoch"] + 1
        best_overall = ck["best"]
        best_gan = ck.get("best_gan", best_overall)
        history = ck["history"]
        print(f"Resumed Branch {branch} from epoch {start_ep}/{args.total_epochs}")
    else:
        base_ck = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
        if base_ck["fingerprint"] != fingerprint:
            raise RuntimeError("Split fingerprint mismatch with base checkpoint!")
        model.load_state_dict(base_ck["model"])
        best_overall = base_ck["val"]["all"]["fMSE"]
        print(f"Initialized Branch {branch} from {args.base_ckpt} (val fMSE={best_overall:.2f})")

    (args.run_dir / "config.json").write_text(
        json.dumps(
            {
                "cfg": cfg,
                "fingerprint": fingerprint,
                "source_ckpt": str(args.base_ckpt),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    val_loader = make_dataloader(records, "val", args.batch_size, args.size, args.workers, args.seed)

    for ep in range(start_ep, args.total_epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        tic = time.perf_counter()

        is_d_warmup = ep == (args.start_epoch + 1)
        model.eval() if is_d_warmup else model.train()
        net_d.train()

        tot_l1, tot_gan, tot_d, count = 0.0, 0.0, 0.0, 0
        scale_accum = [0.0] * args.num_d
        n_critic = 2 if ep < (args.start_epoch + 1 + args.n_critic_warmup) else 1

        train_loader = make_dataloader(records, "train", args.batch_size, args.size, args.workers, args.seed, ep)
        bar = tqdm(train_loader, desc=f"Branch {branch} | Ep {ep}/{args.total_epochs}")
        for c, m, y, _ in bar:
            c = c.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            inp = torch.cat([c, m], dim=1)

            for _ in range(n_critic):
                opt_d.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    with torch.no_grad():
                        raw_fake = model(inp)
                        comp_fake = raw_fake * m + c * (1.0 - m)
                    pred_real = net_d(torch.cat([y, m], dim=1))
                    pred_fake = net_d(torch.cat([comp_fake.detach(), m], dim=1))
                    loss_d, scale_vals = lsgan_discriminator_loss(pred_real, pred_fake, real_label=args.real_label)
                scaler_d.scale(loss_d).backward()
                scaler_d.unscale_(opt_d)
                nn.utils.clip_grad_norm_(net_d.parameters(), 1.0)
                scaler_d.step(opt_d)
                scaler_d.update()

            if is_d_warmup:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    l1_val = masked_l1_loss(raw_fake, y, m)
                gan_val = 0.0
            else:
                opt_g.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    raw = model(inp)
                    comp_out = raw * m + c * (1.0 - m)
                    l1_val = masked_l1_loss(raw, y, m)
                    pred_fake_g = net_d(torch.cat([comp_out, m], dim=1))
                    gan_loss = lsgan_generator_loss(pred_fake_g)
                    loss_g = args.lambda_l1 * l1_val + gan_loss
                    gan_val = float(gan_loss.item())
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(opt_g)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_g.step(opt_g)
                scaler_g.update()

            bs = c.shape[0]
            tot_l1 += float(l1_val.item()) * bs
            tot_gan += gan_val * bs
            tot_d += float(loss_d.item()) * bs
            for idx_s, sv in enumerate(scale_vals):
                scale_accum[idx_s] += sv * bs
            count += bs
            bar.set_postfix(L1=f"{tot_l1 / count:.5f}", GAN=f"{tot_gan / count:.4f}", D=f"{tot_d / count:.4f}")

        scores = evaluate_validation(model, val_loader, device, save_to=args.run_dir / f"val_{ep:03d}.png")
        cur_fmse = scores["all"]["fMSE"]
        improved_overall = cur_fmse < best_overall
        best_overall = min(best_overall, cur_fmse)

        lr_g_used = opt_g.param_groups[0]["lr"]
        lr_d_used = opt_d.param_groups[0]["lr"]
        sched_d.step()
        if not is_d_warmup:
            sched_g.step()

        row = {
            "branch": branch,
            "epoch": ep,
            "train_L1": tot_l1 / count,
            "train_GAN": tot_gan / count,
            "train_D": tot_d / count,
            "val": scores,
            "lr_G": lr_g_used,
            "lr_D": lr_d_used,
            "seconds": time.perf_counter() - tic,
            "peak_VRAM_GB": (torch.cuda.max_memory_allocated() / (2 ** 30)) if device.type == "cuda" else 0.0,
        }
        if args.num_d > 1:
            row["train_D_scales"] = [round(s / count, 5) for s in scale_accum]
        history.append(row)

        payload = {
            "model": model.state_dict(),
            "netD": net_d.state_dict(),
            "cfg": cfg,
            "epoch": ep,
            "fingerprint": fingerprint,
            "val": scores,
        }
        if improved_overall:
            atomic_save(payload, args.run_dir / "best.pt")
        if not is_d_warmup and cur_fmse < best_gan:
            best_gan = cur_fmse
            atomic_save(payload, args.run_dir / "best_gan.pt")

        atomic_save(
            {
                "epoch": ep,
                "model": model.state_dict(),
                "netD": net_d.state_dict(),
                "opt_G": opt_g.state_dict(),
                "opt_D": opt_d.state_dict(),
                "sched_G": sched_g.state_dict(),
                "sched_D": sched_d.state_dict(),
                "scaler_G": scaler_g.state_dict(),
                "scaler_D": scaler_d.state_dict(),
                "cfg": cfg,
                "fingerprint": fingerprint,
                "best": best_overall,
                "best_gan": best_gan,
                "history": history,
                "torch_rng": torch.get_rng_state(),
            },
            last_ckpt,
        )
        (args.run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
