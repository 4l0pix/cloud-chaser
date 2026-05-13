from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from cloud_chaser.data.augmentations import IMAGENET_MEAN, IMAGENET_STD
from cloud_chaser.data.gcd import IMAGE_EXTENSIONS
from cloud_chaser.models.unet import build_unet
from cloud_chaser.utils.checkpoint import load_checkpoint


def _load_unet(weights: str | Path, device: str) -> torch.nn.Module:
    checkpoint = load_checkpoint(weights, map_location=device)
    model = build_unet(
        architecture=checkpoint.get("architecture", "compact"),
        features=checkpoint.get("features"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _unet_boxes(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: str,
    image_size: int,
    threshold: float,
    min_area: int,
) -> list[tuple[int, int, int, int]]:
    h, w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(x.transpose(2, 0, 1))[None].float().to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))[0, 0].float().cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = prob >= threshold
    num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for component_id in range(1, num_labels):
        mask = labels == component_id
        if int(mask.sum()) < min_area:
            continue
        ys, xs = np.where(mask)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def crop_dataset(
    input_root: Path,
    output_root: Path,
    unet_weights: str,
    threshold: float,
    min_area: int,
    padding: int,
    image_size: int,
    device: str,
) -> None:
    model = _load_unet(unet_weights, device)
    paths = sorted(p for p in input_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    for path in tqdm(paths, desc="Cropping cloud regions"):
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes = _unet_boxes(model, image_rgb, device, image_size, threshold, min_area)
        if not boxes:
            rel = path.relative_to(input_root)
            target = output_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), image_bgr)
            continue
        h, w = image_bgr.shape[:2]
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            crop = image_bgr[y1:y2, x1:x2]
            rel = path.relative_to(input_root)
            target = output_root / rel.parent / f"{rel.stem}_cloud{idx}{rel.suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if crop.size > 0:
                cv2.imwrite(str(target), crop)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build classifier crop folders from U-Net cloud components.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--unet-weights", required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-area", type=int, default=256)
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    crop_dataset(
        args.input_root,
        args.output_root,
        args.unet_weights,
        args.threshold,
        args.min_area,
        args.padding,
        args.image_size,
        args.device,
    )


if __name__ == "__main__":
    main()
