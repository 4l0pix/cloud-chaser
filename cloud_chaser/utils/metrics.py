from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    labels = targets.detach().cpu().numpy()
    return {
        "top1": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def binary_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, target).sum() / union)
