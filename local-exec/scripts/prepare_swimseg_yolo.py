from __future__ import annotations

import argparse

from cloud_chaser.config import load_config
from cloud_chaser.data.swimseg import prepare_swimseg_yolo


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SWIMSEG cloud masks to YOLO segmentation format.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    path = prepare_swimseg_yolo(
        root=data_cfg["swimseg_root"],
        output_dir=data_cfg["prepared_seg_dir"],
        val_fraction=data_cfg.get("seg_val_fraction", 0.1),
        test_fraction=data_cfg.get("seg_test_fraction", 0.1),
        seed=cfg["project"]["seed"],
        min_mask_area=data_cfg.get("min_mask_area", 96),
        invert_masks=data_cfg.get("swimseg_invert_masks", False),
    )
    print(f"wrote={path}")


if __name__ == "__main__":
    main()
