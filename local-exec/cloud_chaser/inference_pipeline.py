from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from cloud_chaser.data.augmentations import IMAGENET_MEAN, IMAGENET_STD
from cloud_chaser.data.augmentations import eval_transforms
from cloud_chaser.models.classifier import CloudClassifier
from cloud_chaser.models.unet import build_unet
from cloud_chaser.utils.checkpoint import load_checkpoint
from cloud_chaser.utils.visualization import overlay_instance


def display_class_name(class_name: str) -> str:
    name = class_name.split("_", 1)[1] if "_" in class_name and class_name[0].isdigit() else class_name
    return name.replace("_", " ").title()


@dataclass
class CloudPrediction:
    box: tuple[int, int, int, int]
    segmentation_confidence: float
    class_name: str
    class_confidence: float


class CloudIdentifier:
    def __init__(
        self,
        unet_weights: str | Path,
        classifier_weights: str | Path,
        class_names: list[str] | None = None,
        unet_threshold: float = 0.45,
        unet_min_area: int = 256,
        device: str = "cuda",
        image_size: int = 224,
        half: bool = True,
        crop_padding: int = 12,
    ) -> None:
        self.device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        self.half = half and self.device != "cpu"
        self.crop_padding = crop_padding
        self.image_size = image_size
        self.unet_threshold = unet_threshold
        self.unet_min_area = unet_min_area
        checkpoint = load_checkpoint(unet_weights, map_location=self.device)
        self.unet = build_unet(
            architecture=checkpoint.get("architecture", "compact"),
            features=checkpoint.get("features"),
        ).to(self.device)
        self.unet.load_state_dict(checkpoint["model"])
        self.unet.eval()
        self.transform = eval_transforms(image_size)

        classifier_path = Path(classifier_weights)
        if classifier_path.suffix in {".torchscript", ".ts"}:
            if class_names is None:
                raise ValueError("class_names are required when loading a TorchScript classifier.")
            self.classes = class_names
            self.classifier = torch.jit.load(str(classifier_path), map_location=self.device).to(self.device)
        else:
            checkpoint = load_checkpoint(classifier_path, map_location=self.device)
            self.classes = checkpoint["classes"]
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
        for _, box_tensor in zip(masks, boxes, strict=False):
            x1, y1, x2, y2 = [int(v) for v in box_tensor.tolist()]
            x1 = max(0, x1 - self.crop_padding)
            y1 = max(0, y1 - self.crop_padding)
            x2 = min(w, x2 + self.crop_padding)
            y2 = min(h, y2 + self.crop_padding)
            crop = image_rgb[y1:y2, x1:x2]
            crops.append(self.transform(image=crop)["image"])
        return crops

    def _unet_instances(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        h, w = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        tensor = torch.from_numpy(x.transpose(2, 0, 1))[None].float().to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.half):
            prob = torch.sigmoid(self.unet(tensor))[0, 0].detach().float().cpu().numpy()
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = prob >= self.unet_threshold
        num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        masks: list[np.ndarray] = []
        boxes: list[list[float]] = []
        scores: list[float] = []
        for component_id in range(1, num_labels):
            mask = labels == component_id
            area = int(mask.sum())
            if area < self.unet_min_area:
                continue
            ys, xs = np.where(mask)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            masks.append(mask)
            boxes.append([x1, y1, x2, y2])
            scores.append(float(prob[mask].mean()))
        if not masks:
            return torch.empty((0, h, w), dtype=torch.bool), torch.empty((0, 4)), []
        return torch.from_numpy(np.stack(masks)).bool(), torch.tensor(boxes, dtype=torch.float32), scores

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> tuple[np.ndarray, list[CloudPrediction]]:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        masks, boxes, segmentation_scores = self._unet_instances(image_rgb)

        if len(boxes) == 0:
            return image_bgr, []

        crops = self._crop_instances(image_rgb, masks, boxes)
        batch = torch.stack(crops).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.half):
            probs = torch.softmax(self.classifier(batch), dim=1)
        confs, labels = probs.max(dim=1)

        overlay = image_bgr.copy()
        predictions: list[CloudPrediction] = []
        for i, (box_tensor, seg_score, label_idx, cls_score) in enumerate(
            zip(boxes, segmentation_scores, labels, confs, strict=False)
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
                    segmentation_confidence=float(seg_score),
                    class_name=class_name,
                    class_confidence=class_conf,
                )
            )
        return overlay, predictions
