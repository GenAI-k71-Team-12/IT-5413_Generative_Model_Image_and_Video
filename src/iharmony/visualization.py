"""Visualization utilities for multi-model image comparison grids and quantitative charts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from iharmony.data.dataset import ARCHIVES, GROUP_ORDER, HarmonyTestDataset
from iharmony.models.registry import infer_raw_and_out, load_harmonization_model


def render_comparison_grid(
    chosen_rows: list[dict[str, str]],
    models_spec: dict[str, dict],
    device: torch.device,
    save_path: Path,
    group_meta: Optional[dict[str, dict]] = None,
    title_prefix: str = "",
    img_size: int = 256,
    show: bool = False,
) -> Path:
    """Render Input | Mask | All Available Models | Ground Truth comparison grid."""
    active_keys = [
        k
        for k, info in models_spec.items()
        if info["arch"] != "identity" and info["path"] is not None and Path(info["path"]).is_file()
    ]
    if not active_keys:
        raise RuntimeError("No available model checkpoints found for comparison grid.")

    ds_vis = HarmonyTestDataset(chosen_rows, size=img_size)
    cs, ms, ys, _, _ = zip(*[ds_vis[i] for i in range(len(ds_vis))])
    c_b = torch.stack(cs).to(device)
    m_b = torch.stack(ms).to(device)
    y_b = torch.stack(ys).to(device)

    outs = {}
    ep_labels = {}
    for key in active_keys:
        info = models_spec[key]
        net, meta = load_harmonization_model(info["arch"], info["path"], device)
        _, out = infer_raw_and_out(net, info["arch"], c_b, m_b, use_amp=info["amp"])
        outs[key] = out.cpu()
        ep_labels[key] = f"Ep {meta['epoch']}" if isinstance(meta["epoch"], int) else str(meta["epoch"])
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cols_tensors = [c_b.cpu(), m_b.expand_as(c_b).cpu()] + [outs[k] for k in active_keys] + [y_b.cpu()]
    col_titles = (
        ["Input\nComposite", "Mask"]
        + [f"{models_spec[k]['short']}\n({ep_labels[k]})" for k in active_keys]
        + ["Ground\nTruth"]
    )

    ncols = len(col_titles)
    nrows = len(chosen_rows)
    panels = torch.stack(cols_tensors, dim=1).flatten(0, 1)

    padding = 4
    grid = make_grid(panels, nrow=ncols, padding=padding, pad_value=1.0)

    fig, ax = plt.subplots(figsize=(ncols * 2.7, nrows * 2.7 + 1.4))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")

    step_x = img_size + padding
    for idx, title in enumerate(col_titles):
        ax.text(
            padding + idx * step_x + img_size / 2,
            -12,
            title,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#111111",
        )

    step_y = img_size + padding
    for row_idx, r in enumerate(chosen_rows):
        g_tag = (group_meta or {}).get(r["full_id"], {}).get("joint_group", "")
        lbl = f"{r['subset']}\n{g_tag}" if g_tag else f"{r['subset']}\n#{row_idx + 1}"
        ax.text(
            -12,
            padding + row_idx * step_y + img_size / 2,
            lbl,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, fontweight="bold", y=1.01)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def plot_evaluation_charts(
    all_results: dict[str, dict],
    models_spec: dict[str, dict],
    charts_dir: Path,
    has_groups: bool = False,
    show: bool = False,
) -> list[Path]:
    """Generate dataset bar charts, difficulty-group heatmap, and efficiency scatter plot."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    model_names = [k for k in all_results if all_results[k]["_meta"]["arch"] != "identity"]
    short_labels = [models_spec[k]["short"].replace("\n", " ") for k in model_names]
    cmap = plt.get_cmap("tab10")

    # 1. Grouped bar charts by sub-dataset
    datasets_plot = list(ARCHIVES) + ["all"]
    ds_labels = list(ARCHIVES) + ["ALL"]
    x = np.arange(len(datasets_plot))
    width = 0.8 / max(len(model_names), 1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for idx, (mname, slbl) in enumerate(zip(model_names, short_labels)):
        fmse_vals = [all_results[mname][d]["fMSE"] for d in datasets_plot]
        psnr_vals = [all_results[mname][d]["PSNR"] for d in datasets_plot]
        offset = (idx - (len(model_names) - 1) / 2.0) * width
        axes[0].bar(x + offset, fmse_vals, width, label=slbl, color=cmap(idx % 10))
        axes[1].bar(x + offset, psnr_vals, width, label=slbl, color=cmap(idx % 10))

    axes[0].set_title("fMSE by Sub-dataset (Lower is Better)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ds_labels, fontweight="bold")
    axes[0].set_ylabel("fMSE (0..255 scale)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_title("PSNR by Sub-dataset (Higher is Better)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ds_labels, fontweight="bold")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_ylim(bottom=28)
    axes[1].grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(model_names)), bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    p1 = charts_dir / "metrics_by_dataset.png"
    plt.savefig(p1, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    saved.append(p1)

    # 2. Heatmap across 6 difficulty groups
    if has_groups:
        mat = np.array(
            [[all_results[m]["by_joint_group"][g]["fMSE"] for g in GROUP_ORDER] for m in model_names]
        )
        fig, ax = plt.subplots(figsize=(11, max(4.5, len(model_names) * 0.65)))
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(np.arange(len(GROUP_ORDER)))
        ax.set_xticklabels(GROUP_ORDER, fontsize=10, fontweight="bold")
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_yticklabels(short_labels, fontsize=10, fontweight="bold")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if val > (mat.min() + 0.6 * (mat.max() - mat.min())) else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=9.5, fontweight="bold")
        ax.set_title("fMSE Heatmap across 6 Difficulty Groups (mask_size x color_shift)", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, label="fMSE")
        plt.tight_layout()
        p2 = charts_dir / "fmse_heatmap_difficulty_groups.png"
        plt.savefig(p2, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        plt.close(fig)
        saved.append(p2)

    # 3. Efficiency trade-off scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (mname, slbl) in enumerate(zip(model_names, short_labels)):
        m = all_results[mname]["_meta"]
        fmse = all_results[mname]["all"]["fMSE"]
        lat = m["ms_per_img"]
        params = m["total_params_M"]
        ax.scatter(lat, fmse, s=max(params * 18, 80), alpha=0.82, color=cmap(idx % 10), edgecolors="black", linewidth=1.0)
        ax.annotate(f"{slbl} ({params:.1f}M)", (lat, fmse), textcoords="offset points", xytext=(8, 4), fontsize=9, fontweight="bold")
    ax.set_xlabel("GPU Inference Latency (ms / image)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Test fMSE on 7,404 images", fontsize=11, fontweight="bold")
    ax.set_title("Quality vs Speed & Model Size Trade-off", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = charts_dir / "tradeoff_fmse_speed_params.png"
    plt.savefig(p3, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    saved.append(p3)

    return saved
