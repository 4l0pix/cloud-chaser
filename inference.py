from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from cloud_chaser.config import get_device, load_config
from cloud_chaser.inference_pipeline import CloudIdentifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cloud segmentation and type classification.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    inf_cfg = cfg["inference"]
    det_cfg = cfg["detector"]
    identifier = CloudIdentifier(
        detector_weights=inf_cfg["detector_weights"],
        classifier_weights=inf_cfg["classifier_weights"],
        device=get_device(cfg),
        image_size=data_cfg["image_size"],
        detector_conf=det_cfg["conf"],
        detector_iou=det_cfg["iou"],
        half=det_cfg["half"],
        crop_padding=inf_cfg["crop_padding"],
    )
    overlay, predictions = identifier.predict(args.image)
    output = Path(args.output) if args.output else Path(inf_cfg["output_dir"]) / Path(args.image).name
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), overlay)
    for pred in predictions:
        print(
            f"{pred.class_name}: class={pred.class_confidence:.3f} "
            f"detector={pred.detector_confidence:.3f} box={pred.box}"
        )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
