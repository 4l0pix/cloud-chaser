from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cloud_chaser.data.augmentations import IMAGENET_MEAN, IMAGENET_STD
from cloud_chaser.data.gcd import IMAGE_EXTENSIONS

MASK_TOKENS = (
    "mask",
    "masks",
    "gt",
    "gtmaps",
    "groundtruth",
    "ground_truth",
    "truth",
    "label",
    "labels",
    "annotation",
    "annotations",
    "segmentation",
    "binary",
)


@dataclass(frozen=True)
class SwimsegPair:
    image_path: Path
    mask_path: Path


def _is_probable_binary_mask(path: Path) -> bool:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    sample = image[:: max(1, image.shape[0] // 256), :: max(1, image.shape[1] // 256)]
    unique = np.unique(sample)
    if len(unique) <= 8:
        return True
    low_high = ((sample < 16) | (sample > 239)).mean()
    return bool(low_high > 0.97)


def _looks_like_mask_path(path: Path) -> bool:
    joined = "/".join(part.lower() for part in path.parts)
    return any(token in joined for token in MASK_TOKENS) or _is_probable_binary_mask(path)


def _normal_stem(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(
        r"(_|-)?(mask|gt|gtmap|groundtruth|ground_truth|truth|label|annotation|segmentation|binary)$",
        "",
        stem,
    )
    return re.sub(r"[^a-z0-9]+", "", stem)


def discover_swimseg_pairs(root: str | Path) -> list[SwimsegPair]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"SWIMSEG root does not exist: {root}")

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise RuntimeError(f"No image files found under {root}")

    mask_set = {p for p in files if _looks_like_mask_path(p)}
    mask_files = sorted(mask_set)
    image_files = sorted(p for p in files if p not in mask_set)

    if not image_files or not mask_files:
        mask_set = {p for p in files if _is_probable_binary_mask(p)}
        mask_files = sorted(mask_set)
        image_files = sorted(p for p in files if p not in mask_set)

    masks_by_key: dict[str, list[Path]] = {}
    for mask in mask_files:
        masks_by_key.setdefault(_normal_stem(mask), []).append(mask)

    pairs: list[SwimsegPair] = []
    used_masks: set[Path] = set()
    for image in image_files:
        candidates = [m for m in masks_by_key.get(_normal_stem(image), []) if m not in used_masks]
        if candidates:
            mask = candidates[0]
            pairs.append(SwimsegPair(image, mask))
            used_masks.add(mask)

    # Some packaged SWIMSEG variants keep images and masks aligned by sorted order.
    if len(pairs) < min(len(image_files), len(mask_files)) * 0.5:
        pairs = []
        for image, mask in zip(sorted(image_files), sorted(mask_files), strict=False):
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            msk = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            if img is not None and msk is not None and img.shape[:2] == msk.shape[:2]:
                pairs.append(SwimsegPair(image, mask))

    if not pairs:
        raise RuntimeError(
            f"Could not pair SWIMSEG images and masks under {root}. "
            "Inspect the dataset tree and update data.swimseg_root."
        )
    print(f"Discovered {len(pairs)} SWIMSEG image/mask pairs under {root}")
    return pairs


def _binary_cloud_mask(mask_path: Path, invert: bool = False) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L"))
    binary = mask > 127
    if invert:
        binary = ~binary
    return binary.astype(np.uint8)


class SwimsegMaskDataset(Dataset):
    def __init__(
        self,
        prepared_dir: str | Path,
        split: str,
        image_size: int,
    ) -> None:
        self.prepared_dir = Path(prepared_dir)
        self.split = split
        self.image_size = image_size
        manifest = self.prepared_dir / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"SWIMSEG manifest not found: {manifest}")
        self.records: list[tuple[Path, Path]] = []
        with manifest.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.records.append((Path(row["image"]), Path(row["mask"])))
        if not self.records:
            raise RuntimeError(f"No SWIMSEG records found for split={split} in {manifest}")
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.records[index]
        image = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Could not read SWIMSEG pair: {image_path}, {mask_path}")
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        mask = (mask > 127).astype(np.float32)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask[None]).float()
        return image_tensor, mask_tensor


def _safe_link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        target.symlink_to(source.resolve())
    except OSError:
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(source)
        cv2.imwrite(str(target), image)


def prepare_swimseg_masks(
    root: str | Path,
    output_dir: str | Path,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    invert_masks: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    pairs = discover_swimseg_pairs(root)
    indices = list(range(len(pairs)))
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=val_fraction + test_fraction,
        random_state=seed,
    )
    relative_test = test_fraction / max(val_fraction + test_fraction, 1e-9)
    val_idx, test_idx = train_test_split(holdout_idx, test_size=relative_test, random_state=seed)
    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}

    manifest_path = output_dir / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "mask"])
        writer.writeheader()
        for split, split_idx in split_indices.items():
            for idx in tqdm(split_idx, desc=f"Preparing SWIMSEG {split}"):
                pair = pairs[idx]
                image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                h, w = image.shape[:2]
                binary = _binary_cloud_mask(pair.mask_path, invert=invert_masks)
                if binary.shape != (h, w):
                    binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)

                target_image = output_dir / "images" / split / pair.image_path.name
                target_mask = output_dir / "masks" / split / f"{pair.image_path.stem}.png"
                _safe_link_or_copy(pair.image_path, target_image)
                target_mask.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target_mask), binary * 255)
                writer.writerow({"split": split, "image": str(target_image), "mask": str(target_mask)})

    return manifest_path
