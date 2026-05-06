from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from cloud_chaser.utils.metrics import binary_iou


def evaluate_detector(cfg: dict) -> dict[str, float]:
    """Evaluate YOLO mask metrics and foreground mIoU on the SWIMSEG validation split."""
    from ultralytics import YOLO

    data_cfg = cfg["data"]
    det_cfg = cfg["detector"]
    weights = cfg["inference"]["detector_weights"]
    model = YOLO(weights)
    data_yaml = Path(data_cfg["prepared_seg_dir"]) / "cloud_seg.yaml"
    map_metrics = model.val(
        data=str(data_yaml),
        task="segment",
        imgsz=det_cfg["imgsz"],
        conf=det_cfg["conf"],
        iou=det_cfg["iou"],
        half=det_cfg["half"],
        verbose=False,
    )

    image_dir = Path(data_cfg["prepared_seg_dir"]) / "images" / "val"
    mask_dir = Path(data_cfg["prepared_seg_dir"]) / "masks" / "val"
    ious: list[float] = []
    try:
        for image_path in tqdm(sorted(image_dir.glob("*")), desc="SWIMSEG validation mIoU"):
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                continue
            target = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            result = model.predict(
                image,
                conf=det_cfg["conf"],
                iou=det_cfg["iou"],
                retina_masks=True,
                verbose=False,
            )[0]
            pred = np.zeros(target.shape, dtype=bool)
            if result.masks is not None:
                masks = result.masks.data.detach().cpu().numpy()
                for mask in masks:
                    if mask.shape != target.shape:
                        mask = cv2.resize(mask.astype("float32"), target.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    pred |= mask > 0.5
            ious.append(binary_iou(pred, target))
    except RuntimeError as exc:
        print(f"Skipping custom SWIMSEG mIoU loop after Ultralytics validation: {exc}")

    metrics = {
        "mask_map50_95": float(getattr(map_metrics.seg, "map", 0.0)),
        "mask_map50": float(getattr(map_metrics.seg, "map50", 0.0)),
        "miou": float(np.mean(ious)) if ious else float("nan"),
    }
    print(metrics)
    return metrics
