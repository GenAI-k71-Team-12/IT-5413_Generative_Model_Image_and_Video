"""Shared utilities for reproducibility, checkpoint I/O, and archive extraction."""

from __future__ import annotations

import os
import random
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def atomic_save(obj: object, target: Path) -> None:
    """Write a PyTorch checkpoint atomically via a temporary sibling file."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, target)


def strip_module_prefix(state_dict: dict) -> dict:
    """Remove leading 'module.' prefix from DataParallel/DDP state dicts if present."""
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
    return state_dict


def resolve_run_file(rel_path: str, candidate_runs_dirs: Sequence[Path]) -> Path:
    """Return the first existing checkpoint path across candidate runs directories."""
    for base in candidate_runs_dirs:
        p = Path(base) / rel_path
        if p.is_file():
            return p
    return Path(candidate_runs_dirs[0]) / rel_path


def extract_iharmony4_zips(
    zip_dir: Path,
    data_dir: Path,
    subsets: Iterable[str] = ("HCOCO", "HAdobe5k", "HFlickr", "Hday2night"),
) -> Path:
    """Extract resized 256x256 iHarmony4 zip archives if not already extracted."""
    zip_dir = Path(zip_dir)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for subset in subsets:
        dest = data_dir / subset
        marker = dest / ".extracted_ok"
        if not marker.exists():
            zip_path = zip_dir / f"{subset}_256.zip"
            if not zip_path.is_file():
                raise FileNotFoundError(f"Archive not found: {zip_path}")
            print(f"Extracting {zip_path.name} -> {data_dir} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
            marker.touch()
    return data_dir
