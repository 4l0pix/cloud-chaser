from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from cloud_chaser.config import get_device, load_config
from cloud_chaser.data.gcd import build_gcd_records
from cloud_chaser.inference_pipeline import CloudIdentifier, display_class_name
from cloud_chaser.utils.seed import seed_everything


def _is_clear_sky(class_name: str) -> bool:
    normalized = class_name.lower().replace("_", "").replace("-", "")
    return "clearsky" in normalized or normalized.endswith("clear") or "clear" in normalized


def _best_image_prediction(predictions) -> tuple[str | None, float]:
    if not predictions:
        return None, 0.0
    best = max(predictions, key=lambda p: p.segmentation_confidence * p.class_confidence)
    return best.class_name, best.segmentation_confidence * best.class_confidence


def _report_path(output_dir: Path, prefix: str, suffix: str) -> Path:
    return output_dir / f"{prefix}_{suffix}"


def _evaluate_cascade(
    cfg: dict,
    output_dir: Path,
    samples: int,
    prefix: str,
) -> tuple[dict, list[dict]]:
    records, classes = build_gcd_records(
        cfg["data"]["gcd_root"],
        split="val",
        classes=cfg["data"].get("classification_classes"),
        val_fraction=cfg["data"]["val_fraction"],
        seed=cfg["project"]["seed"],
    )
    clear_sky = [_is_clear_sky(name) for name in classes]
    identifier = CloudIdentifier(
        unet_weights=cfg["inference"]["unet_weights"],
        classifier_weights=cfg["inference"]["classifier_weights"],
        class_names=classes,
        unet_threshold=cfg.get("unet", {}).get("threshold", 0.45),
        unet_min_area=cfg.get("unet", {}).get("min_area", 256),
        device=get_device(cfg),
        image_size=cfg["data"]["image_size"],
        half=cfg.get("unet", {}).get("half", True),
        crop_padding=cfg["inference"]["crop_padding"],
    )

    n = len(classes)
    total_by_class = np.zeros(n, dtype=np.int64)
    segmentation_gate_correct_by_class = np.zeros(n, dtype=np.int64)
    detected_by_class = np.zeros(n, dtype=np.int64)
    classified_correct_by_class = np.zeros(n, dtype=np.int64)
    classified_total_by_class = np.zeros(n, dtype=np.int64)
    cascade_correct_by_class = np.zeros(n, dtype=np.int64)
    confusion = np.zeros((n, n), dtype=np.int64)

    details: list[dict] = []
    display_to_idx = {display_class_name(name): idx for idx, name in enumerate(classes)}

    for record in tqdm(records, desc="GCD cascade validation"):
        true_idx = int(record.label)
        true_name = display_class_name(classes[true_idx])
        expects_cloud = not clear_sky[true_idx]
        _, predictions = identifier.predict(record.path)
        has_detection = len(predictions) > 0
        pred_name, pred_score = _best_image_prediction(predictions)
        pred_idx = display_to_idx.get(pred_name) if pred_name is not None else None

        segmentation_gate_correct = has_detection if expects_cloud else not has_detection
        classification_correct = pred_idx == true_idx if has_detection and pred_idx is not None else False
        if expects_cloud:
            cascade_correct = has_detection and classification_correct
        else:
            cascade_correct = not has_detection

        total_by_class[true_idx] += 1
        detected_by_class[true_idx] += int(has_detection)
        segmentation_gate_correct_by_class[true_idx] += int(segmentation_gate_correct)
        cascade_correct_by_class[true_idx] += int(cascade_correct)
        if has_detection and pred_idx is not None:
            classified_total_by_class[true_idx] += 1
            classified_correct_by_class[true_idx] += int(classification_correct)
            confusion[true_idx, pred_idx] += 1

        details.append(
            {
                "path": str(record.path),
                "true_class": true_name,
                "expects_cloud": expects_cloud,
                "has_detection": has_detection,
                "segmentation_gate_correct": bool(segmentation_gate_correct),
                "predicted_class": pred_name,
                "prediction_score": float(pred_score),
                "classification_correct": bool(classification_correct),
                "cascade_correct": bool(cascade_correct),
                "num_detections": len(predictions),
            }
        )

    segmentation_gate_accuracy = float(segmentation_gate_correct_by_class.sum() / max(total_by_class.sum(), 1))
    conditional_classification_accuracy = float(
        classified_correct_by_class.sum() / max(classified_total_by_class.sum(), 1)
    )
    cascade_accuracy = float(cascade_correct_by_class.sum() / max(total_by_class.sum(), 1))

    metrics = {
        "split": "val",
        "segmenter": "unet",
        "num_images": int(total_by_class.sum()),
        "classes": classes,
        "note": (
            "GCD has image-level class labels but no cloud masks. U-Net segmentation is evaluated as an "
            "image-level cascade gate: non-clearsky classes should produce at least one cloud "
            "detection, while clearsky should produce none."
        ),
        "segmentation_gate_accuracy": segmentation_gate_accuracy,
        "classifier_accuracy_given_detection": conditional_classification_accuracy,
        "cascade_accuracy": cascade_accuracy,
        "total_by_class": total_by_class.tolist(),
        "detected_by_class": detected_by_class.tolist(),
        "segmentation_gate_correct_by_class": segmentation_gate_correct_by_class.tolist(),
        "classified_total_by_class": classified_total_by_class.tolist(),
        "classified_correct_by_class": classified_correct_by_class.tolist(),
        "cascade_correct_by_class": cascade_correct_by_class.tolist(),
        "confusion_matrix_given_detection": confusion.tolist(),
        "details": details,
    }
    _report_path(output_dir, prefix, "metrics.json").write_text(json.dumps(metrics, indent=2))
    _plot_cascade_bars(metrics, output_dir, prefix)
    overlay_path = _make_overlay_grid(cfg, output_dir, details, samples, prefix)
    return metrics, details if overlay_path else details


def _plot_cascade_bars(metrics: dict, output_dir: Path, prefix: str) -> None:
    labels = [display_class_name(name) for name in metrics["classes"]]
    total = np.array(metrics["total_by_class"])
    segmentation_gate_correct = np.array(metrics["segmentation_gate_correct_by_class"])
    classified_total = np.array(metrics["classified_total_by_class"])
    classified_correct = np.array(metrics["classified_correct_by_class"])
    cascade_correct = np.array(metrics["cascade_correct_by_class"])

    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(13, 6.5))
    b1 = ax.bar(x - width, segmentation_gate_correct, width, color="#457b9d", label="U-Net gate correct")
    b2 = ax.bar(x, classified_correct, width, color="#2a9d8f", label="Classified correct after detection")
    b3 = ax.bar(x + width, cascade_correct, width, color="#e76f51", label="End-to-end correct")
    ax.set_title(
        "GCD validation cascade: U-Net cloud segmentation -> cloud type classification\n"
        f"seg-gate={metrics['segmentation_gate_accuracy']:.1%}, "
        f"cls|det={metrics['classifier_accuracy_given_detection']:.1%}, "
        f"end-to-end={metrics['cascade_accuracy']:.1%}"
    )
    ax.set_ylabel("Images")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for bars, denominators in [(b1, total), (b2, classified_total), (b3, total)]:
        for bar, denom in zip(bars, denominators, strict=False):
            value = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(total.max() * 0.015, 1),
                f"{value}/{int(denom)}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    fig.tight_layout()
    fig.savefig(_report_path(output_dir, prefix, "bar.png"), dpi=180)
    plt.close(fig)


def _select_samples(details: list[dict], samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups = [
        [d for d in details if d["has_detection"] and d["cascade_correct"]],
        [d for d in details if d["has_detection"] and not d["cascade_correct"]],
        [d for d in details if not d["has_detection"] and d["expects_cloud"]],
        [d for d in details if not d["has_detection"] and not d["expects_cloud"]],
    ]
    selected: list[dict] = []
    for group in groups:
        rng.shuffle(group)
        selected.extend(group[: max(1, samples // len(groups))])
    if len(selected) < samples:
        remaining = [d for d in details if d not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: samples - len(selected)])
    return selected[:samples]


def _make_overlay_grid(
    cfg: dict,
    output_dir: Path,
    details: list[dict],
    samples: int,
    prefix: str,
) -> Path | None:
    selected = _select_samples(details, samples, cfg["project"]["seed"])
    if not selected:
        return None

    classes = cfg["data"].get("classification_classes")
    identifier = CloudIdentifier(
        unet_weights=cfg["inference"]["unet_weights"],
        classifier_weights=cfg["inference"]["classifier_weights"],
        class_names=classes,
        unet_threshold=cfg.get("unet", {}).get("threshold", 0.45),
        unet_min_area=cfg.get("unet", {}).get("min_area", 256),
        device=get_device(cfg),
        image_size=cfg["data"]["image_size"],
        half=cfg.get("unet", {}).get("half", True),
        crop_padding=cfg["inference"]["crop_padding"],
    )

    overlays: list[np.ndarray] = []
    titles: list[str] = []
    for item in tqdm(selected, desc="GCD cascade overlay samples"):
        overlay_bgr, predictions = identifier.predict(item["path"])
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        if predictions:
            pred = ", ".join(f"{p.class_name} {p.class_confidence:.0%}" for p in predictions[:2])
        else:
            pred = "No cloud detection"
        status = "OK" if item["cascade_correct"] else "FAIL"
        overlays.append(overlay_rgb)
        titles.append(f"{status} | True: {item['true_class']}\nPred: {pred}")

    cols = min(3, len(overlays))
    rows = math.ceil(len(overlays) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.6 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes_array.ravel():
        ax.axis("off")
    for ax, image, title in zip(axes_array.ravel(), overlays, titles, strict=False):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    output_path = _report_path(output_dir, prefix, "overlay_samples.jpg")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the GCD U-Net segmentation/classification cascade.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or "gcd_val_unet_cascade"

    metrics, _ = _evaluate_cascade(cfg, output_dir, args.samples, prefix)
    print("segmenter=unet")
    print(f"saved={_report_path(output_dir, prefix, 'bar.png')}")
    print(f"saved={_report_path(output_dir, prefix, 'overlay_samples.jpg')}")
    print(f"saved={_report_path(output_dir, prefix, 'metrics.json')}")
    print(f"GCD U-Net segmentation gate accuracy={metrics['segmentation_gate_accuracy']:.4f}")
    print(f"GCD classifier accuracy given detection={metrics['classifier_accuracy_given_detection']:.4f}")
    print(f"GCD cascade accuracy={metrics['cascade_accuracy']:.4f}")


if __name__ == "__main__":
    main()
