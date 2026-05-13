from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from cloud_chaser.config import load_config
from cloud_chaser.training import evaluate_unet_segmenter, train_unet_segmenter
from scripts.gcd_visual_report import _evaluate_cascade


def _variant_cfg(base_cfg: dict, variant: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["unet"]["architecture"] = variant["architecture"]
    cfg["unet"]["features"] = variant["features"]
    checkpoint = Path(cfg["project"]["output_dir"]) / "unet" / variant["name"] / "best.pt"
    cfg["unet"]["checkpoint"] = str(checkpoint)
    cfg["inference"]["unet_weights"] = str(checkpoint)
    return cfg


def _plot_summary(rows: list[dict], output_dir: Path) -> Path:
    labels = [row["name"] for row in rows]
    miou = [float(row["swimseg_miou"]) for row in rows]
    dice = [float(row["swimseg_dice"]) for row in rows]
    cascade = [float(row["cascade_accuracy"]) for row in rows]

    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = list(range(len(rows)))
    width = 0.25
    ax.bar([i - width for i in x], miou, width, label="SWIMSEG mIoU")
    ax.bar(x, dice, width, label="SWIMSEG Dice")
    ax.bar([i + width for i in x], cascade, width, label="GCD cascade")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Six U-Net pipeline comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v:.2f}" for v in container.datavalues], fontsize=7, padding=2)
    fig.tight_layout()
    output_path = output_dir / "unet_ablation_summary.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate the six U-Net research pipelines.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="../results/reports/unet_ablation")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for variant in base_cfg["experiments"]["unet_variants"]:
        name = variant["name"]
        cfg = _variant_cfg(base_cfg, variant)
        if not args.skip_train:
            train_unet_segmenter(cfg, experiment_name=name, unet_override=variant)
        unet_metrics = evaluate_unet_segmenter(cfg, checkpoint_path=cfg["unet"]["checkpoint"], unet_override=variant)
        prefix = f"{name}_gcd"
        cascade_metrics, _ = _evaluate_cascade(cfg, output_dir, args.samples, prefix)
        row = {
            "name": name,
            "architecture": variant["architecture"],
            "features": " ".join(str(v) for v in variant["features"]),
            "checkpoint": cfg["unet"]["checkpoint"],
            "swimseg_loss": unet_metrics["loss"],
            "swimseg_miou": unet_metrics["miou"],
            "swimseg_dice": unet_metrics["dice"],
            "segmentation_gate_accuracy": cascade_metrics["segmentation_gate_accuracy"],
            "classifier_accuracy_given_detection": cascade_metrics["classifier_accuracy_given_detection"],
            "cascade_accuracy": cascade_metrics["cascade_accuracy"],
        }
        rows.append(row)
        (output_dir / f"{name}_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    summary_path = output_dir / "unet_ablation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = _plot_summary(rows, output_dir)
    print(f"saved={summary_path}")
    print(f"saved={plot_path}")


if __name__ == "__main__":
    main()
