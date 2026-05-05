from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from cloud_chaser.data.gcd import IMAGE_EXTENSIONS


def crop_dataset(input_root: Path, output_root: Path, detector_weights: str, conf: float, padding: int) -> None:
    model = YOLO(detector_weights)
    paths = sorted(p for p in input_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    for path in tqdm(paths, desc="Cropping cloud regions"):
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = model.predict(image_rgb, conf=conf, retina_masks=True, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            rel = path.relative_to(input_root)
            target = output_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), image_bgr)
            continue
        h, w = image_bgr.shape[:2]
        masks = result.masks.data.detach().cpu().numpy() if result.masks is not None else []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            crop = image_bgr.copy()
            if len(masks):
                mask = masks[idx]
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype("float32"), (w, h), interpolation=cv2.INTER_NEAREST)
                crop[mask <= 0.5] = 0
            crop = crop[y1:y2, x1:x2]
            rel = path.relative_to(input_root)
            target = output_root / rel.parent / f"{rel.stem}_cloud{idx}{rel.suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if np.prod(crop.shape[:2]) > 0:
                cv2.imwrite(str(target), crop)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build classifier crop folders from detector masks.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--detector-weights", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--padding", type=int, default=12)
    args = parser.parse_args()
    crop_dataset(args.input_root, args.output_root, args.detector_weights, args.conf, args.padding)


if __name__ == "__main__":
    main()
