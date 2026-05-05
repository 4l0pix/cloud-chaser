from __future__ import annotations

from pathlib import Path


def train_yolo_segmenter(
    model_name_or_path: str,
    data_yaml: str | Path,
    output_dir: str | Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    patience: int = 20,
    lr0: float = 0.01,
    weight_decay: float = 0.0005,
):
    from ultralytics import YOLO

    output_dir = Path(output_dir)
    yolo = YOLO(model_name_or_path)
    return yolo.train(
        data=str(data_yaml),
        task="segment",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(output_dir.parent),
        name=output_dir.name,
        patience=patience,
        lr0=lr0,
        weight_decay=weight_decay,
        amp=True,
    )
