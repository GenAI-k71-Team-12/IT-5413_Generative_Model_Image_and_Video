#!/usr/bin/env python3
"""Unified evaluation CLI for all 8 models in runs/ (+ Input Composite baseline) on iHarmony4 test (7,404 images)."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from iharmony.data import (
    ARCHIVES,
    GROUP_ORDER,
    HarmonyTestDataset,
    build_records,
    load_difficulty_groups,
)
from iharmony.metrics import METRIC_KEYS, aggregate_metrics_df, compute_batch_metrics
from iharmony.models import build_default_models_spec, infer_raw_and_out, load_harmonization_model
from iharmony.visualization import plot_evaluation_charts, render_comparison_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate all Image Harmonization models in runs/ on iHarmony4 test set.")
    p.add_argument("--data-dir", type=Path, required=True, help="Extracted iHarmony4 256x256 directory.")
    p.add_argument("--runs-dir", type=Path, required=True, help="Directory containing model run folders.")
    p.add_argument("--fallback-runs-dir", type=Path, default=None, help="Optional secondary runs directory.")
    p.add_argument("--groups-csv", type=Path, default=None, help="Path to iHarmony4_groups_train_test/test.csv.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: <runs-dir>/evaluation_all_models).")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--include-30ep", action="store_true", help="Also evaluate Masked L1 30-epoch milestone.")
    p.add_argument("--save-back-to-runs", action="store_true", help="Save test_metrics_unified.json into each run folder.")
    p.add_argument("--no-visuals", action="store_true", help="Skip generating visual comparison grids.")
    p.add_argument("--no-charts", action="store_true", help="Skip generating matplotlib summary charts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or (args.runs_dir / "evaluation_all_models")
    for sub in ("per_image_metrics", "previews", "charts"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    candidate_dirs = [args.runs_dir] + ([args.fallback_runs_dir] if args.fallback_runs_dir else [])
    models_spec = build_default_models_spec(candidate_dirs, include_masked_l1_30ep=args.include_30ep)

    records, fingerprint, _ = build_records(args.data_dir, seed=42, val_fraction=0.05)
    group_meta = load_difficulty_groups(args.groups_csv)
    print(f"Test set verified: {len(records['test'])} images | fingerprint={fingerprint}")
    if group_meta:
        print(f"Loaded difficulty groups: {len(group_meta)} entries from {args.groups_csv}")

    test_ds = HarmonyTestDataset(records["test"], size=256)
    loaders: dict[int, DataLoader] = {}

    def get_loader(bs: int) -> DataLoader:
        if bs not in loaders:
            loaders[bs] = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=args.workers, pin_memory=True)
        return loaders[bs]

    all_results: dict[str, dict] = {}
    for name, info in models_spec.items():
        ck_path = info["path"]
        if ck_path is not None and not Path(ck_path).is_file():
            print(f"[SKIP] Missing checkpoint: {ck_path}")
            continue

        model, meta = load_harmonization_model(info["arch"], ck_path, device, expected_fp=fingerprint)
        loader = get_loader(info["batch_size"])

        if model is not None:
            dummy_c = torch.zeros(2, 3, 256, 256, device=device)
            dummy_m = torch.ones(2, 1, 256, 256, device=device)
            _ = infer_raw_and_out(model, info["arch"], dummy_c, dummy_m, use_amp=info["amp"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            del dummy_c, dummy_m

        rows_out = []
        saved_preview = False
        total_infer_sec = 0.0
        total_imgs = 0

        for c, m, y, subsets, full_ids in tqdm(loader, desc=f"{name} (Ep {meta['epoch']})"):
            c = c.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            raw, out = infer_raw_and_out(model, info["arch"], c, m, use_amp=info["amp"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_infer_sec += time.perf_counter() - t0
            total_imgs += c.shape[0]

            vals = compute_batch_metrics(out, y, m)
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

            if not saved_preview and ck_path is not None:
                panels = torch.stack([c, m.expand_as(c), raw, out, y], dim=1).flatten(0, 1)
                save_image(panels[:30], str(out_dir / "previews" / f"{info['slug']}_preview.png"), nrow=5)
                if args.save_back_to_runs:
                    ep_tag = f"ep{meta['epoch']}" if isinstance(meta["epoch"], int) else "pretrained"
                    save_image(panels[:30], str(Path(ck_path).parent / f"test_preview_{ep_tag}_unified.png"), nrow=5)
                saved_preview = True

        df = pd.DataFrame(rows_out)
        df.to_csv(out_dir / "per_image_metrics" / f"{info['slug']}.csv", index=False)

        res: dict = {"all": aggregate_metrics_df(df)}
        for sub in ARCHIVES:
            sub_df = df[df["subset"] == sub]
            if len(sub_df) > 0:
                res[sub] = aggregate_metrics_df(sub_df)
        if group_meta:
            res["by_mask_group"] = {
                g: aggregate_metrics_df(df[df["mask_group"] == g])
                for g in ("small", "large")
                if (df["mask_group"] == g).any()
            }
            res["by_color_group"] = {
                g: aggregate_metrics_df(df[df["color_group"] == g])
                for g in ("low", "medium", "high")
                if (df["color_group"] == g).any()
            }
            res["by_joint_group"] = {
                g: aggregate_metrics_df(df[df["joint_group"] == g])
                for g in GROUP_ORDER
                if (df["joint_group"] == g).any()
            }

        ms_per_img = (total_infer_sec / max(total_imgs, 1)) * 1000.0
        res["_meta"] = {
            **meta,
            "slug": info["slug"],
            "arch": info["arch"],
            "checkpoint": str(ck_path) if ck_path is not None else None,
            "batch_size": info["batch_size"],
            "amp": info["amp"],
            "ms_per_img": round(ms_per_img, 3),
            "fps": round(1000.0 / max(ms_per_img, 1e-6), 1),
        }
        all_results[name] = res

        if args.save_back_to_runs and ck_path is not None:
            (Path(ck_path).parent / "test_metrics_unified.json").write_text(
                json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (out_dir / "all_models_test_metrics.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    base_fmse = all_results.get("0. Input Composite (No Harm)", {}).get("all", {}).get("fMSE")
    overall_rows = []
    for name, res in all_results.items():
        m, a = res["_meta"], res["all"]
        p_str = (
            f"{m['trainable_params_M']:.1f}/{m['total_params_M']:.1f}M"
            if m["trainable_params_M"] != m["total_params_M"]
            else f"{m['total_params_M']:.2f}M"
        )
        gain_pct = ((base_fmse - a["fMSE"]) / base_fmse * 100.0) if base_fmse else None
        overall_rows.append(
            {
                "Model": name,
                "Ep": m["epoch"],
                "Params": p_str,
                "ms/img": m["ms_per_img"],
                "MSE": a["MSE"],
                "fMSE": a["fMSE"],
                "fMSE_Gain_%": round(gain_pct, 2) if gain_pct is not None else None,
                "PSNR": a["PSNR"],
                "fPSNR": a["fPSNR"],
                "SSIM": a["SSIM"],
                "fSSIM": a["fSSIM"],
                "fDeltaE76": a["fDeltaE76"],
            }
        )
    pd.DataFrame(overall_rows).to_csv(out_dir / "summary_overall.csv", index=False)

    sub_rows = [
        {"Model": name, "Dataset": sub.upper() if sub == "all" else sub, **res[sub]}
        for name, res in all_results.items()
        for sub in list(ARCHIVES) + ["all"]
        if sub in res
    ]
    pd.DataFrame(sub_rows).to_csv(out_dir / "summary_by_dataset.csv", index=False)

    if group_meta:
        grp_rows = [
            {"Model": name, "Joint_Group": g, **res["by_joint_group"][g]}
            for name, res in all_results.items()
            for g in GROUP_ORDER
            if g in res.get("by_joint_group", {})
        ]
        pd.DataFrame(grp_rows).to_csv(out_dir / "summary_by_difficulty_group.csv", index=False)

    if not args.no_visuals:
        rng = random.Random(123)
        chosen_random = []
        for subset in ARCHIVES:
            sub_rows_t = [r for r in records["test"] if r["subset"] == subset]
            chosen_random.extend(rng.sample(sub_rows_t, min(2, len(sub_rows_t))))
        render_comparison_grid(
            chosen_random,
            models_spec,
            device,
            out_dir / "comparison_all_models_4datasets.png",
            group_meta=group_meta,
            title_prefix="All Models Visual Comparison (2 random samples / sub-dataset)",
        )
        if group_meta:
            chosen_hard = []
            for subset in ARCHIVES:
                hard_rows = [
                    r
                    for r in records["test"]
                    if r["subset"] == subset and group_meta.get(r["full_id"], {}).get("joint_group") == "large_high"
                ]
                hard_rows.sort(key=lambda r: group_meta[r["full_id"]]["input_deltaE76"], reverse=True)
                if len(hard_rows) >= 2:
                    chosen_hard.extend([hard_rows[len(hard_rows) // 10], hard_rows[len(hard_rows) // 3]])
                else:
                    chosen_hard.extend(hard_rows)
            if chosen_hard:
                render_comparison_grid(
                    chosen_hard,
                    models_spec,
                    device,
                    out_dir / "comparison_all_models_hardest_large_high.png",
                    group_meta=group_meta,
                    title_prefix="Hardest Group Comparison (large_high)",
                )

    if not args.no_charts:
        plot_evaluation_charts(all_results, models_spec, out_dir / "charts", has_groups=bool(group_meta))

    print(f"Evaluation complete. All artifacts saved under: {out_dir}")


if __name__ == "__main__":
    main()
