from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from cloud_chaser.config import get_device, load_config
from cloud_chaser.models.classifier import CloudClassifier
from cloud_chaser.models.unet import build_unet
from cloud_chaser.utils.checkpoint import load_checkpoint


def _export_torch_model(model: torch.nn.Module, dummy: torch.Tensor, output_path: Path, fmt: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "torchscript":
        traced = torch.jit.trace(model, dummy)
        traced.save(str(output_path))
    elif fmt == "onnx":
        if importlib.util.find_spec("onnxscript") is None:
            print(f"Skipping ONNX export for {output_path.name}: onnxscript is not installed.")
            return output_path
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["image"],
            output_names=["output"],
            dynamic_axes={"image": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
    else:
        raise ValueError("Format must be 'torchscript' or 'onnx'")
    return output_path


def export_unet(cfg: dict, fmt: str, output: str | None = None) -> Path:
    device = get_device(cfg)
    checkpoint_path = cfg["unet"]["checkpoint"]
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = build_unet(
        architecture=checkpoint.get("architecture", cfg["unet"].get("architecture", "compact")),
        features=checkpoint.get("features", cfg["unet"].get("features")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    image_size = cfg["data"]["image_size"]
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    output_path = Path(output or Path(checkpoint_path).with_suffix(f".{fmt}"))
    return _export_torch_model(model, dummy, output_path, fmt)


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
    return _export_torch_model(model, dummy, output_path, fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export U-Net segmenter or classifier artifacts.")
    parser.add_argument("target", choices=["unet", "classifier"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.target == "unet":
        output = export_unet(cfg, args.format, args.output)
        print(f"saved={output}")
    else:
        output = export_classifier(cfg, args.format, args.output)
        print(f"saved={output}")


if __name__ == "__main__":
    main()
