from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from cloud_chaser.data.ade20k import _iter_pairs, resolve_ade_classes
from cloud_chaser.utils.metrics import binary_iou


def evaluate_detector(cfg: dict) -> dict[str, float]:
    """Evaluate YOLO mask metrics and ADE validation mIoU for the cloud/sky foreground."""
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

    root = Path(data_cfg["ade20k_root"])
    class_map = resolve_ade_classes(
        root,
        data_cfg["ade_classes"],
        data_cfg.get("ade_fallback_class_ids", []),
    )
    ious: list[float] = []
    for image_path, mask_path in tqdm(_iter_pairs(root, "validation"), desc="ADE20K validation mIoU"):
        target = np.isin(np.array(Image.open(mask_path)), class_map.class_ids)
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

    metrics = {
        "mask_map50_95": float(getattr(map_metrics.seg, "map", 0.0)),
        "mask_map50": float(getattr(map_metrics.seg, "map50", 0.0)),
        "miou": float(np.mean(ious)) if ious else 0.0,
    }
    print(metrics)
    return metrics
