from __future__ import annotations

import argparse

from cloud_chaser.config import load_config
from cloud_chaser.data.ade20k import prepare_ade20k_yolo


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ADE20K cloud/sky masks to YOLO segmentation format.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    path = prepare_ade20k_yolo(
        root=data_cfg["ade20k_root"],
        output_dir=data_cfg["prepared_seg_dir"],
        class_names=data_cfg["ade_classes"],
        fallback_class_ids=data_cfg.get("ade_fallback_class_ids", []),
        min_mask_area=data_cfg["min_mask_area"],
    )
    print(f"wrote={path}")


if __name__ == "__main__":
    main()
