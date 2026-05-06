from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloud_chaser.config import get_device
from cloud_chaser.data.augmentations import (
    classification_train_transforms,
    eval_transforms,
    ssl_transforms,
)
from cloud_chaser.data.gcd import GCDDataset
from cloud_chaser.data.swimseg import prepare_swimseg_yolo
from cloud_chaser.data.tjnu import UnlabeledCloudDataset
from cloud_chaser.models.classifier import CloudClassifier
from cloud_chaser.models.detector import train_yolo_segmenter
from cloud_chaser.models.simclr import SimCLR, nt_xent_loss
from cloud_chaser.utils.checkpoint import load_checkpoint, save_checkpoint
from cloud_chaser.utils.metrics import classification_metrics
from cloud_chaser.utils.seed import seed_everything


def train_detector(cfg: dict) -> None:
    seed_everything(cfg["project"]["seed"])
    data_cfg = cfg["data"]
    detector_cfg = cfg["detector"]
    data_yaml = prepare_swimseg_yolo(
        root=data_cfg["swimseg_root"],
        output_dir=data_cfg["prepared_seg_dir"],
        val_fraction=data_cfg.get("seg_val_fraction", 0.1),
        test_fraction=data_cfg.get("seg_test_fraction", 0.1),
        seed=cfg["project"]["seed"],
        min_mask_area=data_cfg.get("min_mask_area", 96),
        invert_masks=data_cfg.get("swimseg_invert_masks", False),
    )
    output_dir = Path(cfg["project"]["output_dir"]) / "detector"
    train_yolo_segmenter(
        model_name_or_path=detector_cfg["model"],
        data_yaml=data_yaml,
        output_dir=output_dir,
        epochs=detector_cfg["epochs"],
        imgsz=detector_cfg["imgsz"],
        batch=detector_cfg["batch"],
        device=get_device(cfg),
        patience=detector_cfg["patience"],
        lr0=detector_cfg["lr0"],
        weight_decay=detector_cfg["weight_decay"],
    )


def train_ssl(cfg: dict) -> None:
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)
    data_cfg = cfg["data"]
    ssl_cfg = cfg["ssl"]
    tjnu_root = Path(data_cfg["tjnu_root"])
    if not tjnu_root.exists():
        print(f"Skipping optional SimCLR pretraining: TJNU dataset not found at {tjnu_root}")
        print("Classifier training can continue from an ImageNet-pretrained backbone instead.")
        return
    dataset = UnlabeledCloudDataset(
        tjnu_root,
        ssl_transforms(data_cfg["image_size"], cfg["augmentation"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=ssl_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=device == "cuda",
        drop_last=True,
    )
    model = SimCLR(
        backbone=ssl_cfg["backbone"],
        projection_dim=ssl_cfg["projection_dim"],
        hidden_dim=ssl_cfg["hidden_dim"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ssl_cfg["lr"],
        weight_decay=ssl_cfg["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    best_loss = float("inf")
    output_dir = Path(cfg["project"]["output_dir"]) / "ssl"

    for epoch in range(ssl_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        for view1, view2 in tqdm(loader, desc=f"SSL epoch {epoch + 1}/{ssl_cfg['epochs']}"):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = nt_xent_loss(model(view1), model(view2), ssl_cfg["temperature"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
        epoch_loss = running_loss / max(1, len(loader))
        save_checkpoint(
            {
                "epoch": epoch,
                "backbone": ssl_cfg["backbone"],
                "encoder": model.encoder.state_dict(),
                "model": model.state_dict(),
                "loss": epoch_loss,
            },
            output_dir / "last.pt",
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "backbone": ssl_cfg["backbone"],
                    "encoder": model.encoder.state_dict(),
                    "model": model.state_dict(),
                    "loss": epoch_loss,
                },
                output_dir / "best.pt",
            )


def _run_classifier_epoch(
    model: CloudClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = True,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    scaler = torch.cuda.amp.GradScaler(enabled=training and amp and device == "cuda")
    for images, targets in tqdm(loader, desc="train" if training else "eval"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu())
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    metrics = classification_metrics(logits, targets)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def train_classifier(cfg: dict) -> None:
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)
    data_cfg = cfg["data"]
    cls_cfg = cfg["classifier"]
    train_tfms = classification_train_transforms(data_cfg["image_size"], **cfg["augmentation"])
    eval_tfms = eval_transforms(data_cfg["image_size"])
    train_ds = GCDDataset(
        data_cfg["gcd_root"],
        "train",
        train_tfms,
        classes=data_cfg["classification_classes"],
        val_fraction=data_cfg["val_fraction"],
        seed=cfg["project"]["seed"],
    )
    val_ds = GCDDataset(
        data_cfg["gcd_root"],
        "val",
        eval_tfms,
        classes=train_ds.classes,
        val_fraction=data_cfg["val_fraction"],
        seed=cfg["project"]["seed"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cls_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cls_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=device == "cuda",
    )
    model = CloudClassifier(
        num_classes=len(train_ds.classes),
        backbone=cls_cfg["backbone"],
        dropout=cls_cfg["dropout"],
    ).to(device)
    if cls_cfg.get("ssl_checkpoint"):
        checkpoint = load_checkpoint(cls_cfg["ssl_checkpoint"], map_location=device)
        model.encoder.load_state_dict(checkpoint["encoder"], strict=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cls_cfg["lr"], weight_decay=cls_cfg["weight_decay"])
    best_f1 = -1.0
    output_dir = Path(cfg["project"]["output_dir"]) / "classifier"

    for epoch in range(cls_cfg["epochs"]):
        model.freeze_encoder(epoch < cls_cfg.get("freeze_backbone_epochs", 0))
        train_metrics = _run_classifier_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, amp=cls_cfg["amp"]
        )
        val_metrics = _run_classifier_epoch(model, val_loader, criterion, device, amp=cls_cfg["amp"])
        print(
            f"epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} "
            f"val_top1={val_metrics['top1']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )
        payload = {
            "epoch": epoch,
            "classes": train_ds.classes,
            "backbone": cls_cfg["backbone"],
            "model": model.state_dict(),
            "val_metrics": val_metrics,
        }
        save_checkpoint(payload, output_dir / "last.pt")
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            save_checkpoint(payload, output_dir / "best.pt")


def evaluate_classifier(cfg: dict) -> dict[str, float]:
    device = get_device(cfg)
    data_cfg = cfg["data"]
    cls_cfg = cfg["classifier"]
    checkpoint = load_checkpoint(cls_cfg["checkpoint"], map_location=device)
    dataset = GCDDataset(
        data_cfg["gcd_root"],
        "test",
        eval_transforms(data_cfg["image_size"]),
        classes=checkpoint["classes"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cls_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=device == "cuda",
    )
    model = CloudClassifier(
        num_classes=len(checkpoint["classes"]),
        backbone=checkpoint["backbone"],
        dropout=0.0,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = _run_classifier_epoch(model, loader, nn.CrossEntropyLoss(), device, amp=cls_cfg["amp"])
    print(metrics)
    return metrics
