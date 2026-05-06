from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from cloud_chaser.config import save_yaml


@dataclass(frozen=True)
class AdeClassMap:
    requested_names: list[str]
    class_ids: list[int]
    missing_names: list[str]


def parse_object_info(path: str | Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            idx = int(parts[0].strip())
            names = [n.strip().lower() for n in parts[3].split(",") if n.strip()]
            for name in names:
                mapping[name] = idx
    return mapping


def resolve_ade_classes(
    root: str | Path,
    class_names: Iterable[str],
    fallback_ids: Iterable[int] | None = None,
) -> AdeClassMap:
    requested = [name.lower().strip() for name in class_names]
    info_path = Path(root) / "objectInfo150.txt"
    name_to_id = parse_object_info(info_path)
    class_ids: list[int] = []
    missing: list[str] = []
    for name in requested:
        if name in name_to_id:
            class_ids.append(name_to_id[name])
        else:
            missing.append(name)
    for idx in fallback_ids or []:
        if idx not in class_ids:
            class_ids.append(int(idx))
    if not class_ids:
        raise ValueError(
            f"None of the requested ADE classes {requested} were found in {info_path}, "
            "and no fallback IDs were supplied."
        )
    return AdeClassMap(requested_names=requested, class_ids=sorted(set(class_ids)), missing_names=missing)


def _iter_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = root / "images" / split
    mask_dir = root / "annotations" / split
    pairs: list[tuple[Path, Path]] = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        stem = mask_path.stem
        image_path = image_dir / f"{stem}.jpg"
        if image_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def _mask_to_yolo_polygons(mask: np.ndarray, min_area: int) -> list[list[float]]:
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    polygons: list[list[float]] = []
    h, w = mask.shape[:2]
    for component_id in range(1, num_labels):
        component = (labels == component_id).astype(np.uint8)
        if int(component.sum()) < min_area:
            continue
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or len(contour) < 3:
                continue
            epsilon = 0.0025 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            coords: list[float] = []
            for x, y in approx:
                coords.extend([float(np.clip(x / w, 0, 1)), float(np.clip(y / h, 0, 1))])
            if len(coords) >= 6:
                polygons.append(coords)
    return polygons


def prepare_ade20k_yolo(
    root: str | Path,
    output_dir: str | Path,
    class_names: Iterable[str],
    fallback_class_ids: Iterable[int] | None = None,
    min_mask_area: int = 512,
) -> Path:
    """Convert ADE20K semantic masks into a one-class YOLO segmentation dataset."""
    root = Path(root)
    output_dir = Path(output_dir)
    class_map = resolve_ade_classes(root, class_names, fallback_class_ids)
    split_map = {"training": "train", "validation": "val"}

    for ade_split, yolo_split in split_map.items():
        image_out = output_dir / "images" / yolo_split
        label_out = output_dir / "labels" / yolo_split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        pairs = _iter_pairs(root, ade_split)
        for image_path, mask_path in tqdm(pairs, desc=f"Preparing ADE20K {ade_split}"):
            ade_mask = np.array(Image.open(mask_path))
            binary = np.isin(ade_mask, class_map.class_ids)
            polygons = _mask_to_yolo_polygons(binary, min_area=min_mask_area)
            if not polygons:
                continue
            target_image = image_out / image_path.name
            if not target_image.exists():
                target_image.symlink_to(image_path.resolve())
            label_path = label_out / f"{image_path.stem}.txt"
            lines = ["0 " + " ".join(f"{v:.6f}" for v in polygon) for polygon in polygons]
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "cloud"},
        "metadata": {
            "ade_class_ids": class_map.class_ids,
            "requested_ade_classes": class_map.requested_names,
            "missing_requested_classes": class_map.missing_names,
        },
    }
    save_yaml(data_yaml, output_dir / "cloud_seg.yaml")
    return output_dir / "cloud_seg.yaml"
