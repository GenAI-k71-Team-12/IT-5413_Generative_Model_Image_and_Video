"""Deterministic iHarmony4 indexing, SHA-256 split fingerprinting, difficulty groups, and Datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

ARCHIVES = ("HCOCO", "HAdobe5k", "HFlickr", "Hday2night")
GROUP_ORDER = ("small_low", "small_medium", "small_high", "large_low", "large_medium", "large_high")
EXTS = {".jpg", ".jpeg", ".png"}


def make_index(folder: Path) -> dict[str, Path]:
    """Map file stem -> Path for image files in folder."""
    return {
        p.stem: p
        for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith(".")
    }


def build_records(
    data_dir: Path,
    seed: int = 42,
    val_fraction: float = 0.05,
    subsets: Iterable[str] = ARCHIVES,
) -> tuple[dict[str, list[dict[str, str]]], str, dict[str, int]]:
    """Build deterministic Train/Val/Test records grouped by background source_id.

    Returns:
        (records, sha256_fingerprint, official_counts)
    """
    data_dir = Path(data_dir)
    records: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    official_counts: dict[str, int] = {}

    for subset in subsets:
        root = data_dir / subset
        comp = make_index(root / "composite_images")
        mask = make_index(root / "masks")
        real = make_index(root / "real_images")

        subset_rows: dict[str, list[dict[str, str]]] = {}
        for split in ("train", "test"):
            txt = root / f"{subset}_{split}.txt"
            rows: list[dict[str, str]] = []
            for line in txt.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                stem = Path(line.strip().replace("\\", "/").split()[0]).stem
                parts = stem.split("_")
                source_id = "_".join(parts[:-2])
                mask_id = "_".join(parts[:-1])
                rows.append(
                    {
                        "subset": subset,
                        "stem": stem,
                        "group": f"{subset}/{source_id}",
                        "full_id": f"{subset}/{stem}",
                        "comp": str(comp[stem]),
                        "mask": str(mask[mask_id]),
                        "real": str(real[source_id]),
                    }
                )
            subset_rows[split] = rows
            official_counts[f"{subset}_{split}"] = len(rows)

        groups = sorted({r["group"] for r in subset_rows["train"]})
        random.Random(seed).shuffle(groups)
        val_groups = set(groups[: max(1, round(len(groups) * val_fraction))])

        for r in subset_rows["train"]:
            records["val" if r["group"] in val_groups else "train"].append(r)
        records["test"].extend(subset_rows["test"])

    split_ids = {s: [r["full_id"] for r in rows] for s, rows in records.items()}
    fingerprint = hashlib.sha256(json.dumps(split_ids, sort_keys=True).encode()).hexdigest()

    if set(split_ids["train"]) & set(split_ids["test"]):
        raise RuntimeError("Data leakage detected between train and test splits!")
    if set(split_ids["val"]) & set(split_ids["test"]):
        raise RuntimeError("Data leakage detected between val and test splits!")

    return records, fingerprint, official_counts


def load_difficulty_groups(csv_path: Optional[Path]) -> dict[str, dict[str, object]]:
    """Load 6-way difficulty group metadata from iHarmony4_groups_train_test/test.csv."""
    if csv_path is None:
        return {}
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return {}

    group_meta: dict[str, dict[str, object]] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("sample_key") or f"{row['dataset']}/{row['sample_id']}"
            group_meta[key] = {
                "mask_group": row["mask_group"],
                "color_group": row["color_group"],
                "joint_group": row["joint_group"],
                "mask_ratio": float(row["mask_ratio"]),
                "input_deltaE76": float(row["input_deltaE76"]),
            }
    return group_meta


def _apply_fg_color_aug(comp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Random brightness/contrast/channel-gain perturbation applied only inside foreground mask."""
    gain = torch.empty(3, 1, 1).uniform_(0.85, 1.15)
    bias = torch.empty(3, 1, 1).uniform_(-0.08, 0.08)
    aug = (comp * gain + bias).clamp(0.0, 1.0)
    return aug * mask + comp * (1.0 - mask)


class HarmonyDataset(Dataset):
    """iHarmony4 dataset for training and validation."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        size: int = 256,
        train: bool = False,
        fg_color_aug_prob: float = 0.0,
    ):
        self.rows = rows
        self.size = size
        self.train = train
        self.fg_color_aug_prob = fg_color_aug_prob

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        target_size = (self.size, self.size)
        imgs = []
        for key, mode in (("comp", "RGB"), ("mask", "L"), ("real", "RGB")):
            with Image.open(r[key]) as im:
                im = im.convert(mode)
                if im.size != target_size:
                    resample = Image.Resampling.NEAREST if key == "mask" else Image.Resampling.BILINEAR
                    im = im.resize(target_size, resample)
                imgs.append(pil_to_tensor(im).float() / 255.0)
        c, m, y = imgs
        m_bin = (m >= 0.5).float()
        m = m_bin if m_bin.sum() > 0 else (m > 0).float()

        if self.train:
            if torch.rand(()) < 0.5:
                c, m, y = c.flip(-1), m.flip(-1), y.flip(-1)
            if self.fg_color_aug_prob > 0.0 and torch.rand(()) < self.fg_color_aug_prob:
                c = _apply_fg_color_aug(c, m)
        return c, m, y, r["subset"]


class HarmonyTestDataset(Dataset):
    """iHarmony4 test dataset yielding (comp, mask, real, subset, full_id)."""

    def __init__(self, rows: list[dict[str, str]], size: int = 256):
        self.rows = rows
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        target_size = (self.size, self.size)
        imgs = []
        for key, mode in (("comp", "RGB"), ("mask", "L"), ("real", "RGB")):
            with Image.open(r[key]) as im:
                im = im.convert(mode)
                if im.size != target_size:
                    resample = Image.Resampling.NEAREST if key == "mask" else Image.Resampling.BILINEAR
                    im = im.resize(target_size, resample)
                imgs.append(pil_to_tensor(im).float() / 255.0)
        c, m, y = imgs
        m_bin = (m >= 0.5).float()
        m = m_bin if m_bin.sum() > 0 else (m > 0).float()
        return c, m, y, r["subset"], r["full_id"]


def make_dataloader(
    records: dict[str, list[dict[str, str]]],
    split: str,
    batch_size: int = 32,
    size: int = 256,
    workers: int = 2,
    seed: int = 42,
    epoch: int = 0,
    fg_color_aug_prob: float = 0.0,
) -> DataLoader:
    """Create a reproducible DataLoader matching the notebook training pipeline."""
    is_train = split == "train"
    gen = torch.Generator().manual_seed(seed + epoch)
    ds = HarmonyDataset(
        records[split],
        size=size,
        train=is_train,
        fg_color_aug_prob=fg_color_aug_prob if is_train else 0.0,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(workers > 0),
        generator=gen,
    )
