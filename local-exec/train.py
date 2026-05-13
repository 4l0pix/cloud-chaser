from __future__ import annotations

import argparse

from cloud_chaser.config import load_config
from cloud_chaser.training import (
    evaluate_classifier,
    evaluate_unet_segmenter,
    train_classifier,
    train_classifier_ssl,
    train_unet_segmenter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud Chaser training entrypoint")
    parser.add_argument(
        "stage",
        choices=["unet", "classifier-ssl", "classifier", "eval-classifier", "eval-unet"],
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.stage == "unet":
        train_unet_segmenter(cfg)
    elif args.stage == "classifier-ssl":
        train_classifier_ssl(cfg)
    elif args.stage == "classifier":
        train_classifier(cfg)
    elif args.stage == "eval-classifier":
        evaluate_classifier(cfg)
    elif args.stage == "eval-unet":
        evaluate_unet_segmenter(cfg)


if __name__ == "__main__":
    main()
