#!/usr/bin/env python3
"""CLI script to extract, normalize, resize (256x256), and optionally repackage iHarmony4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iharmony.data.preprocess import (
    extract_raw_archives,
    normalize_hcoco_layout,
    package_resized_subsets,
    resize_dataset,
    verify_dataset_structure,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess raw iHarmony4 archives into 256x256 dataset.")
    p.add_argument("--raw-dir", type=Path, help="Folder containing raw HCOCO/HAdobe5k/HFlickr/Hday2night zip files.")
    p.add_argument("--extract-dir", type=Path, required=True, help="Directory containing extracted full-res iHarmony4.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for resized dataset.")
    p.add_argument("--size", type=int, default=256, help="Target square image size (default: 256).")
    p.add_argument("--workers", type=int, default=8, help="Number of resize worker threads.")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality for resized RGB images.")
    p.add_argument("--package-dir", type=Path, default=None, help="Optional folder to save <subset>_256.zip archives.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw_dir is not None:
        extract_raw_archives(args.raw_dir, args.extract_dir)
    normalize_hcoco_layout(args.extract_dir)
    resize_dataset(
        src_root=args.extract_dir,
        dst_root=args.out_dir,
        size=args.size,
        max_workers=args.workers,
        jpeg_quality=args.quality,
    )
    verify_dataset_structure(args.out_dir, touch_markers=True)
    if args.package_dir is not None:
        zips = package_resized_subsets(args.out_dir, args.package_dir, size=args.size)
        print("Created archives:", [str(z) for z in zips])
    print(f"Done. Resized dataset ready at: {args.out_dir}")


if __name__ == "__main__":
    main()
