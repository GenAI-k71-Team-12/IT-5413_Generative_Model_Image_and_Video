"""Dataset indexing, preprocessing, and PyTorch DataLoader helpers for iHarmony4."""

from iharmony.data.dataset import (
    ARCHIVES,
    GROUP_ORDER,
    HarmonyDataset,
    HarmonyTestDataset,
    build_records,
    load_difficulty_groups,
    make_dataloader,
    make_index,
)
from iharmony.data.preprocess import (
    extract_raw_archives,
    normalize_hcoco_layout,
    package_resized_subsets,
    resize_dataset,
    verify_dataset_structure,
)

__all__ = [
    "ARCHIVES",
    "GROUP_ORDER",
    "HarmonyDataset",
    "HarmonyTestDataset",
    "build_records",
    "load_difficulty_groups",
    "make_dataloader",
    "make_index",
    "extract_raw_archives",
    "normalize_hcoco_layout",
    "package_resized_subsets",
    "resize_dataset",
    "verify_dataset_structure",
]
