"""iHarmony4 raw archive extraction, HCOCO layout normalization, and 256x256 resizing."""

from __future__ import annotations

import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm.auto import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_SUBSETS = ("HCOCO", "HAdobe5k", "HFlickr", "Hday2night")


def extract_raw_archives(raw_dir: Path, extract_root: Path) -> Path:
    """Extract all .zip files from raw_dir into extract_root and copy external .txt split files."""
    raw_dir = Path(raw_dir)
    extract_root = Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(raw_dir.rglob("*.zip"))
    for z_path in zip_files:
        print(f"Extracting {z_path.name} -> {extract_root} ...")
        with zipfile.ZipFile(z_path, "r") as zf:
            zf.extractall(extract_root)

    for txt_path in raw_dir.rglob("*.txt"):
        shutil.copy2(txt_path, extract_root / txt_path.name)

    return extract_root


def normalize_hcoco_layout(extract_root: Path) -> Path:
    """Move top-level HCOCO folders/files under extract_root/HCOCO for uniform subset layout."""
    extract_root = Path(extract_root)
    hcoco_dir = extract_root / "HCOCO"
    hcoco_dir.mkdir(parents=True, exist_ok=True)

    for name in ("composite_images", "masks", "real_images", "HCOCO_train.txt", "HCOCO_test.txt"):
        src = extract_root / name
        dst = hcoco_dir / name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
    return hcoco_dir


def resize_dataset(
    src_root: Path,
    dst_root: Path,
    size: int = 256,
    max_workers: int = 8,
    jpeg_quality: int = 95,
) -> Path:
    """Resize all iHarmony4 images to (size, size): NEAREST for binary masks, BICUBIC for RGB."""
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    target_size = (size, size)

    tasks: list[tuple[Path, Path, bool]] = []
    txt_files: list[tuple[Path, Path]] = []

    for p in src_root.rglob("*"):
        if not p.is_file() or p.name.startswith(".") or "__MACOSX" in p.parts:
            continue
        dst_path = dst_root / p.relative_to(src_root)
        suffix = p.suffix.lower()
        if suffix in IMAGE_EXTS:
            tasks.append((p, dst_path, "masks" in p.parts))
        elif suffix == ".txt":
            txt_files.append((p, dst_path))

    for src_txt, dst_txt in txt_files:
        dst_txt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_txt, dst_txt)

    def _resize_one(item: tuple[Path, Path, bool]) -> None:
        src_path, dst_path, is_mask = item
        if dst_path.exists():
            return
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as img:
            if is_mask:
                img.convert("L").resize(target_size, Image.Resampling.NEAREST).save(dst_path)
            else:
                img.convert("RGB").resize(target_size, Image.Resampling.BICUBIC).save(
                    dst_path, quality=jpeg_quality
                )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(_resize_one, tasks),
                total=len(tasks),
                desc=f"Resizing to {size}x{size}",
                unit="img",
            )
        )

    return dst_root


def package_resized_subsets(
    src_dir: Path,
    out_dir: Path,
    subsets: Iterable[str] = DEFAULT_SUBSETS,
    size: int = 256,
) -> list[Path]:
    """Pack each resized subset into <subset>_<size>.zip for fast cloud/Colab unpacking."""
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for subset in subsets:
        subset_dir = src_dir / subset
        if not subset_dir.is_dir():
            continue
        zip_path = out_dir / f"{subset}_{size}.zip"
        files = [p for p in subset_dir.rglob("*") if p.is_file()]
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for f in tqdm(files, desc=f"Zipping {subset}", unit="file"):
                zf.write(f, f.relative_to(src_dir))
        created.append(zip_path)
    return created


def verify_dataset_structure(
    data_dir: Path, subsets: Iterable[str] = DEFAULT_SUBSETS, touch_markers: bool = True
) -> None:
    """Verify all four iHarmony4 subsets contain composite_images, masks, real_images, and split lists."""
    data_dir = Path(data_dir)
    for subset in subsets:
        dest = data_dir / subset
        if not dest.is_dir():
            raise FileNotFoundError(f"Missing subset directory: {dest}")
        for sub_dir in ("composite_images", "masks", "real_images"):
            if not (dest / sub_dir).is_dir():
                raise FileNotFoundError(f"Missing folder: {dest / sub_dir}")
        for split in ("train", "test"):
            txt_path = dest / f"{subset}_{split}.txt"
            if not txt_path.is_file():
                raise FileNotFoundError(f"Missing split file: {txt_path}")
        if touch_markers:
            (dest / ".extracted_ok").touch()
