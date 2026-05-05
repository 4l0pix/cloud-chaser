from __future__ import annotations

import argparse

from cloud_chaser.config import load_config
from cloud_chaser.evaluation import evaluate_detector
from cloud_chaser.training import evaluate_classifier, train_classifier, train_detector, train_ssl


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud Chaser training entrypoint")
    parser.add_argument("stage", choices=["detector", "ssl", "classifier", "eval-classifier", "eval-detector"])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.stage == "detector":
        train_detector(cfg)
    elif args.stage == "ssl":
        train_ssl(cfg)
    elif args.stage == "classifier":
        train_classifier(cfg)
    elif args.stage == "eval-classifier":
        evaluate_classifier(cfg)
    elif args.stage == "eval-detector":
        evaluate_detector(cfg)


if __name__ == "__main__":
    main()
