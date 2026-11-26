#!/usr/bin/env python3
"""Dedicated RainNet 256x256 audit: Fair Blind test vs official junleen/RainNet protocols.

Compares 4 evaluation protocols on the exact same checkpoint (`RainNet_net_G_last.pth`):
  R1: Blind (fair, raw composite input & background — no ground-truth background leakage)
  R2: GT-BG input (`comp*m + real*(1-m)`) + fair composite output background
  R3: Official junleen/RainNet float32 protocol (`comp*m + real*(1-m)` for both input & bg)
  R4: Official junleen/RainNet `evaluate.py` protocol (GT-BG + `util.tensor2im` uint8 floor)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from iharmony.data import (  # noqa: E402
    ARCHIVES,
    GROUP_ORDER,
    HarmonyTestDataset,
    build_records,
    load_difficulty_groups,
)
from iharmony.metrics import METRIC_KEYS, aggregate_metrics_df, compute_batch_metrics  # noqa: E402
from iharmony.models import load_harmonization_model  # noqa: E402


RAINNET_PROTOCOLS = [
    {
        "name": "R1. RainNet — Blind (Fair, No GT-BG)",
        "slug": "R1_rainnet_blind_fair",
        "gt_bg_input": False,
        "gt_bg_output": False,
        "tensor2im_uint8": False,
        "fair": True,
    },
    {
        "name": "R2. RainNet — GT-BG In + Fair Comp BG",
        "slug": "R2_rainnet_gtbg_in_fair_bg",
        "gt_bg_input": True,
        "gt_bg_output": False,
        "tensor2im_uint8": False,
        "fair": False,
    },
    {
        "name": "R3. RainNet — Official Repo (Float32, GT-BG*)",
        "slug": "R3_rainnet_official_float32",
        "gt_bg_input": True,
        "gt_bg_output": True,
        "tensor2im_uint8": False,
        "fair": False,
    },
    {
        "name": "R4. RainNet — Official evaluate.py (uint8, GT-BG*)",
        "slug": "R4_rainnet_official_evaluate_uint8",
        "gt_bg_input": True,
        "gt_bg_output": True,
        "tensor2im_uint8": True,
        "fair": False,
    },
]


@torch.inference_mode()
def infer_rainnet_mode(
    model: torch.nn.Module,
    c: Tensor,
    m: Tensor,
    y: Tensor,
    gt_bg_input: bool,
    gt_bg_output: bool,
    tensor2im_uint8: bool,
) -> tuple[Tensor, Tensor]:
    """Run RainNet under the requested protocol and return `(out_eval, y_eval)` in `[0, 1]`."""
    c_m11 = (c - 0.5) / 0.5
    y_m11 = (y - 0.5) / 0.5
    c_in_m11 = (c_m11 * m + y_m11 * (1.0 - m)) if gt_bg_input else c_m11

    raw_m11 = model(c_in_m11, m).float()
    bg_m11 = y_m11 if gt_bg_output else c_m11
    out_m11 = raw_m11 * m + bg_m11 * (1.0 - m)

    if tensor2im_uint8:
        out_u8 = ((out_m11 + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        y_u8 = ((y_m11 + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        return out_u8.float() / 255.0, y_u8.float() / 255.0

    out_01 = ((out_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
    return out_01, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RainNet 256x256: Blind fair test vs official GT-BG protocols.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to extracted iHarmony4 256x256 dataset.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to RainNet_net_G_last.pth.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for JSON/CSV audit tables.")
    parser.add_argument("--groups-csv", type=Path, default=None, help="Optional iHarmony4_groups_train_test/test.csv.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "per_image_metrics").mkdir(parents=True, exist_ok=True)

    records, _, _ = build_records(args.data_root, seed=42, val_fraction=0.05)
    group_meta = load_difficulty_groups(args.groups_csv) if args.groups_csv else {}

    ds = HarmonyTestDataset(records["test"], size=256)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model, meta = load_harmonization_model("rainnet", args.checkpoint, device=device)

    all_results: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for proto in RAINNET_PROTOCOLS:
        rows_out = []
        total_sec, total_imgs = 0.0, 0
        for c, m, y, subsets, full_ids in tqdm(loader, desc=proto["name"]):
            c = c.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_eval, y_eval = infer_rainnet_mode(
                model, c, m, y,
                gt_bg_input=proto["gt_bg_input"],
                gt_bg_output=proto["gt_bg_output"],
                tensor2im_uint8=proto["tensor2im_uint8"],
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_sec += time.perf_counter() - t0
            total_imgs += c.shape[0]

            vals = compute_batch_metrics(out_eval, y_eval, m)
            vals_cpu = {k: v.detach().cpu().numpy() for k, v in vals.items()}
            for i in range(c.shape[0]):
                fid = full_ids[i]
                ginfo = group_meta.get(fid, {})
                row = {
                    "full_id": fid,
                    "subset": subsets[i],
                    "mask_group": ginfo.get("mask_group", ""),
                    "color_group": ginfo.get("color_group", ""),
                    "joint_group": ginfo.get("joint_group", ""),
                    "mask_ratio": ginfo.get("mask_ratio", float(m[i].mean().item())),
                }
                for k in METRIC_KEYS:
                    row[k] = float(vals_cpu[k][i])
                rows_out.append(row)

        df = pd.DataFrame(rows_out)
        df.to_csv(args.out_dir / "per_image_metrics" / f"{proto['slug']}.csv", index=False)

        res = {"all": aggregate_metrics_df(df)}
        for sub in ARCHIVES:
            sub_df = df[df["subset"] == sub]
            if len(sub_df) > 0:
                res[sub] = aggregate_metrics_df(sub_df)
        if group_meta:
            res["by_joint_group"] = {
                g: aggregate_metrics_df(df[df["joint_group"] == g])
                for g in GROUP_ORDER
                if (df["joint_group"] == g).any()
            }

        ms_per_img = (total_sec / max(total_imgs, 1)) * 1000.0
        res["_meta"] = {
            **meta,
            "slug": proto["slug"],
            "gt_bg_input": proto["gt_bg_input"],
            "gt_bg_output": proto["gt_bg_output"],
            "tensor2im_uint8": proto["tensor2im_uint8"],
            "fair_blind": proto["fair"],
            "ms_per_img": round(ms_per_img, 3),
        }
        all_results[proto["name"]] = res
        summary_rows.append({
            "Protocol": proto["name"],
            "Mode": "Blind" if proto["fair"] else "GT-BG*",
            "ms/img": round(ms_per_img, 3),
            **{k: res["all"][k] for k in METRIC_KEYS},
        })

    (args.out_dir / "rainnet_audit_metrics.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(args.out_dir / "rainnet_audit_summary.csv", index=False)
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
