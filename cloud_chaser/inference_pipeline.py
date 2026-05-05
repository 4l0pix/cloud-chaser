from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cloud_chaser.data.augmentations import eval_transforms
from cloud_chaser.models.classifier import CloudClassifier
from cloud_chaser.utils.checkpoint import load_checkpoint
from cloud_chaser.utils.visualization import overlay_instance


def display_class_name(class_name: str) -> str:
    name = class_name.split("_", 1)[1] if "_" in class_name and class_name[0].isdigit() else class_name
    return name.replace("_", " ").title()


@dataclass
class CloudPrediction:
    box: tuple[int, int, int, int]
    detector_confidence: float
    class_name: str
    class_confidence: float


class CloudIdentifier:
    def __init__(
        self,
        detector_weights: str | Path,
        classifier_weights: str | Path,
        device: str = "cuda",
        image_size: int = 224,
        detector_conf: float = 0.25,
        detector_iou: float = 0.6,
        half: bool = True,
        crop_padding: int = 12,
    ) -> None:
        from ultralytics import YOLO

        self.device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        self.detector = YOLO(str(detector_weights))
        self.detector_conf = detector_conf
        self.detector_iou = detector_iou
        self.half = half and self.device != "cpu"
        self.crop_padding = crop_padding
        checkpoint = load_checkpoint(classifier_weights, map_location=self.device)
        self.classes = checkpoint["classes"]
        self.transform = eval_transforms(image_size)
        self.classifier = CloudClassifier(
            num_classes=len(self.classes),
            backbone=checkpoint["backbone"],
            dropout=0.0,
            pretrained=False,
        ).to(self.device)
        self.classifier.load_state_dict(checkpoint["model"])
        self.classifier.eval()

    def _crop_instances(
        self,
        image_rgb: np.ndarray,
        masks: torch.Tensor,
        boxes: torch.Tensor,
    ) -> list[torch.Tensor]:
        h, w = image_rgb.shape[:2]
        crops: list[torch.Tensor] = []
        for mask_tensor, box_tensor in zip(masks, boxes, strict=False):
            x1, y1, x2, y2 = [int(v) for v in box_tensor.tolist()]
            x1 = max(0, x1 - self.crop_padding)
            y1 = max(0, y1 - self.crop_padding)
            x2 = min(w, x2 + self.crop_padding)
            y2 = min(h, y2 + self.crop_padding)
            mask = mask_tensor.detach().cpu().numpy().astype(bool)
            masked = image_rgb.copy()
            masked[~mask] = 0
            crop = masked[y1:y2, x1:x2]
            crops.append(self.transform(image=crop)["image"])
        return crops

    @torch.inference_mode()
    def predict(self, image_path: str | Path) -> tuple[np.ndarray, list[CloudPrediction]]:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        results = self.detector.predict(
            image_rgb,
            conf=self.detector_conf,
            iou=self.detector_iou,
            retina_masks=True,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        result = results[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return image_bgr, []

        masks = result.masks.data
        if masks.shape[-2:] != (h, w):
            masks = F.interpolate(masks[:, None].float(), size=(h, w), mode="nearest")[:, 0] > 0.5
        boxes = result.boxes.xyxy.detach().cpu()
        detector_scores = result.boxes.conf.detach().cpu().tolist()

        crops = self._crop_instances(image_rgb, masks, boxes)
        batch = torch.stack(crops).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.half):
            probs = torch.softmax(self.classifier(batch), dim=1)
        confs, labels = probs.max(dim=1)

        overlay = image_bgr.copy()
        predictions: list[CloudPrediction] = []
        for i, (box_tensor, det_score, label_idx, cls_score) in enumerate(
            zip(boxes, detector_scores, labels, confs, strict=False)
        ):
            box = tuple(int(v) for v in box_tensor.tolist())
            class_name = display_class_name(self.classes[int(label_idx)])
            class_conf = float(cls_score.detach().cpu())
            mask = masks[i].detach().cpu().numpy().astype(bool)
            overlay = overlay_instance(
                overlay,
                mask,
                box,
                class_name,
                class_conf,
                color_index=int(label_idx),
            )
            predictions.append(
                CloudPrediction(
                    box=box,
                    detector_confidence=float(det_score),
                    class_name=class_name,
                    class_confidence=class_conf,
                )
            )
        return overlay, predictions
