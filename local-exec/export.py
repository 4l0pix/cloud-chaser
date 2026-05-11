from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from cloud_chaser.config import get_device, load_config
from cloud_chaser.models.classifier import CloudClassifier
from cloud_chaser.utils.checkpoint import load_checkpoint


def export_detector(weights: str, fmt: str, imgsz: int, half: bool) -> None:
    from ultralytics import YOLO

    YOLO(weights).export(format=fmt, imgsz=imgsz, half=half)


def export_classifier(cfg: dict, fmt: str, output: str | None = None) -> Path:
    device = get_device(cfg)
    checkpoint_path = cfg["classifier"]["checkpoint"]
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = CloudClassifier(
        num_classes=len(checkpoint["classes"]),
        backbone=checkpoint["backbone"],
        dropout=0.0,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    image_size = cfg["data"]["image_size"]
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    output_path = Path(output or Path(checkpoint_path).with_suffix(f".{fmt}"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "torchscript":
        traced = torch.jit.trace(model, dummy)
        traced.save(str(output_path))
    elif fmt == "onnx":
        if importlib.util.find_spec("onnxscript") is None:
            print("Skipping classifier ONNX export: onnxscript is not installed.")
            return output_path
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
    else:
        raise ValueError("Classifier format must be 'torchscript' or 'onnx'")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export detector or classifier artifacts.")
    parser.add_argument("target", choices=["detector", "classifier"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "engine"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.target == "detector":
        export_detector(
            cfg["inference"]["detector_weights"],
            args.format,
            cfg["detector"]["imgsz"],
            cfg["detector"]["half"],
        )
    else:
        output = export_classifier(cfg, args.format, args.output)
        print(f"saved={output}")


if __name__ == "__main__":
    main()
