from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloud_chaser.config import get_device, load_config
from cloud_chaser.data.augmentations import eval_transforms
from cloud_chaser.data.gcd import GCDDataset, build_gcd_records
from cloud_chaser.inference_pipeline import CloudIdentifier, display_class_name
from cloud_chaser.models.classifier import CloudClassifier
from cloud_chaser.utils.checkpoint import load_checkpoint
from cloud_chaser.utils.seed import seed_everything


def _load_classifier(cfg: dict, device: str) -> tuple[CloudClassifier, list[str]]:
    checkpoint = load_checkpoint(cfg["classifier"]["checkpoint"], map_location=device)
    classes = checkpoint["classes"]
    model = CloudClassifier(
        num_classes=len(classes),
        backbone=checkpoint["backbone"],
        dropout=0.0,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, classes


@torch.no_grad()
def _predict_gcd_validation(cfg: dict, output_dir: Path) -> dict:
    device = get_device(cfg)
    model, classes = _load_classifier(cfg, device)
    dataset = GCDDataset(
        root=cfg["data"]["gcd_root"],
        split="val",
        transform=eval_transforms(cfg["data"]["image_size"]),
        classes=cfg["data"].get("classification_classes"),
        val_fraction=cfg["data"]["val_fraction"],
        seed=cfg["project"]["seed"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["classifier"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device == "cuda",
    )

    correct_by_class = np.zeros(len(classes), dtype=np.int64)
    total_by_class = np.zeros(len(classes), dtype=np.int64)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)

    for images, labels in tqdm(loader, desc="GCD validation report"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(images)
        preds = logits.argmax(dim=1)
        for true_idx, pred_idx in zip(labels.cpu().numpy(), preds.cpu().numpy(), strict=False):
            total_by_class[int(true_idx)] += 1
            confusion[int(true_idx), int(pred_idx)] += 1
            if int(true_idx) == int(pred_idx):
                correct_by_class[int(true_idx)] += 1

    accuracy = float(correct_by_class.sum() / max(total_by_class.sum(), 1))
    per_class_accuracy = np.divide(
        correct_by_class,
        np.maximum(total_by_class, 1),
        out=np.zeros_like(correct_by_class, dtype=np.float64),
        where=total_by_class > 0,
    )

    metrics = {
        "split": "val",
        "num_images": int(total_by_class.sum()),
        "top1": accuracy,
        "classes": classes,
        "correct_by_class": correct_by_class.tolist(),
        "total_by_class": total_by_class.tolist(),
        "per_class_accuracy": per_class_accuracy.tolist(),
        "confusion_matrix": confusion.tolist(),
    }
    (output_dir / "gcd_val_metrics.json").write_text(json.dumps(metrics, indent=2))

    labels = [display_class_name(name) for name in classes]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, correct_by_class, color="#2a9d8f")
    ax.bar(x, total_by_class - correct_by_class, bottom=correct_by_class, color="#d8dee4")
    ax.set_title(f"GCD validation correctly predicted images by class (top-1={accuracy:.1%})")
    ax.set_ylabel("Images")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(["Correct", "Incorrect"], frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for bar, correct, total in zip(bars, correct_by_class, total_by_class, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            int(total) + max(total_by_class.max() * 0.015, 1),
            f"{int(correct)}/{int(total)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "gcd_val_correct_bar.png", dpi=180)
    plt.close(fig)

    return metrics


def _make_overlay_grid(cfg: dict, output_dir: Path, samples: int) -> Path | None:
    detector_path = Path(cfg["inference"]["detector_weights"])
    classifier_path = Path(cfg["inference"]["classifier_weights"])
    if not detector_path.exists() or not classifier_path.exists():
        print("Skipping overlay grid: missing detector or classifier inference artifact.")
        return None

    records, classes = build_gcd_records(
        cfg["data"]["gcd_root"],
        split="val",
        classes=cfg["data"].get("classification_classes"),
        val_fraction=cfg["data"]["val_fraction"],
        seed=cfg["project"]["seed"],
    )
    rng = random.Random(cfg["project"]["seed"])
    selected = rng.sample(records, k=min(samples, len(records)))

    identifier = CloudIdentifier(
        detector_weights=detector_path,
        classifier_weights=classifier_path,
        class_names=classes,
        device=get_device(cfg),
        image_size=cfg["data"]["image_size"],
        detector_conf=cfg["detector"]["conf"],
        detector_iou=cfg["detector"]["iou"],
        half=cfg["detector"]["half"],
        crop_padding=cfg["inference"]["crop_padding"],
    )

    overlays: list[np.ndarray] = []
    titles: list[str] = []
    for record in tqdm(selected, desc="GCD validation overlay samples"):
        overlay_bgr, predictions = identifier.predict(record.path)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        true_name = display_class_name(classes[record.label])
        pred_text = ", ".join(f"{p.class_name} {p.class_confidence:.0%}" for p in predictions[:2])
        if not pred_text:
            pred_text = "No cloud detection"
        overlays.append(overlay_rgb)
        titles.append(f"True: {true_name}\nPred: {pred_text}")

    if not overlays:
        return None

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
    output_path = output_dir / "gcd_val_overlay_samples.jpg"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create GCD validation charts and overlay samples.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="/kaggle/working/cloud-chaser/reports")
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _predict_gcd_validation(cfg, output_dir)
    overlay_path = _make_overlay_grid(cfg, output_dir, args.samples)
    print(f"saved={output_dir / 'gcd_val_correct_bar.png'}")
    print(f"saved={output_dir / 'gcd_val_metrics.json'}")
    if overlay_path:
        print(f"saved={overlay_path}")
    print(f"GCD validation top1={metrics['top1']:.4f}")


if __name__ == "__main__":
    main()
