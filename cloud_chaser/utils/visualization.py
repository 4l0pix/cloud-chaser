from __future__ import annotations

import cv2
import numpy as np


PALETTE = [
    (42, 157, 143),
    (233, 196, 106),
    (230, 111, 81),
    (38, 70, 83),
    (131, 197, 190),
    (239, 71, 111),
    (17, 138, 178),
]


def overlay_instance(
    image: np.ndarray,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    score: float,
    color_index: int,
    alpha: float = 0.42,
) -> np.ndarray:
    color = PALETTE[color_index % len(PALETTE)]
    out = image.copy()
    color_layer = np.zeros_like(out)
    color_layer[:, :] = color
    out[mask] = cv2.addWeighted(out, 1 - alpha, color_layer, alpha, 0)[mask]
    x1, y1, x2, y2 = box
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    text = f"{label} - {score:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(out, (x1, y_text), (x1 + tw + 8, y_text + th + 8), color, -1)
    cv2.putText(
        out,
        text,
        (x1 + 4, y_text + th + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out
