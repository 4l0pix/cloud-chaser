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
)
from cloud_chaser.data.gcd import GCDDataset
from cloud_chaser.data.swimseg import SwimsegMaskDataset, prepare_swimseg_yolo
from cloud_chaser.models.classifier import CloudClassifier
from cloud_chaser.models.detector import train_yolo_segmenter
from cloud_chaser.models.unet import CloudUNet
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


def _segmentation_scores(logits: torch.Tensor, masks: torch.Tensor) -> tuple[float, float]:
    preds = torch.sigmoid(logits) > 0.5
    targets = masks > 0.5
    intersection = (preds & targets).sum(dim=(1, 2, 3)).float()
    union = (preds | targets).sum(dim=(1, 2, 3)).float().clamp_min(1.0)
    pred_sum = preds.sum(dim=(1, 2, 3)).float()
    target_sum = targets.sum(dim=(1, 2, 3)).float()
    iou = (intersection / union).mean().item()
    dice = ((2 * intersection + 1.0) / (pred_sum + target_sum + 1.0)).mean().item()
    return iou, dice


def _run_unet_epoch(
    model: CloudUNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    scaler = torch.cuda.amp.GradScaler(enabled=training and device == "cuda")
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    for images, masks in tqdm(loader, desc="unet-train" if training else "unet-eval"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(images)
                bce = criterion(logits, masks)
                probs = torch.sigmoid(logits)
                intersection = (probs * masks).sum(dim=(1, 2, 3))
                dice_loss = 1 - ((2 * intersection + 1) / (probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1)).mean()
                loss = bce + dice_loss
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        iou, dice = _segmentation_scores(logits.detach(), masks.detach())
        total_loss += float(loss.detach().cpu())
        total_iou += iou
        total_dice += dice
    count = max(1, len(loader))
    return {"loss": total_loss / count, "miou": total_iou / count, "dice": total_dice / count}


def train_unet_detector(cfg: dict) -> None:
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)
    data_cfg = cfg["data"]
    unet_cfg = cfg["unet"]
    prepare_swimseg_yolo(
        root=data_cfg["swimseg_root"],
        output_dir=data_cfg["prepared_seg_dir"],
        val_fraction=data_cfg.get("seg_val_fraction", 0.1),
        test_fraction=data_cfg.get("seg_test_fraction", 0.1),
        seed=cfg["project"]["seed"],
        min_mask_area=data_cfg.get("min_mask_area", 96),
        invert_masks=data_cfg.get("swimseg_invert_masks", False),
    )
    train_ds = SwimsegMaskDataset(data_cfg["prepared_seg_dir"], "train", data_cfg["image_size"])
    val_ds = SwimsegMaskDataset(data_cfg["prepared_seg_dir"], "val", data_cfg["image_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=unet_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=unet_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=device == "cuda",
    )
    model = CloudUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=unet_cfg["lr"], weight_decay=unet_cfg["weight_decay"])
    output_dir = Path(cfg["project"]["output_dir"]) / "unet"
    last_checkpoint = output_dir / "last.pt"
    best_checkpoint = output_dir / "best.pt"
    best_miou = -1.0
    start_epoch = 0
    resume_checkpoint = last_checkpoint if last_checkpoint.exists() else best_checkpoint if best_checkpoint.exists() else None
    if resume_checkpoint is not None:
        checkpoint = load_checkpoint(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_miou = float(checkpoint.get("best_miou", checkpoint.get("val_metrics", {}).get("miou", -1.0)))
        print(f"Resuming U-Net from {resume_checkpoint} at epoch {start_epoch + 1}")
    for epoch in range(start_epoch, unet_cfg["epochs"]):
        train_metrics = _run_unet_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_unet_epoch(model, val_loader, criterion, device)
        print(
            f"unet_epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} "
            f"val_miou={val_metrics['miou']:.4f} val_dice={val_metrics['dice']:.4f}"
        )
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "best_miou": max(best_miou, val_metrics["miou"]),
        }
        save_checkpoint(payload, output_dir / "last.pt")
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(payload, output_dir / "best.pt")


def evaluate_unet_detector(cfg: dict) -> dict[str, float]:
    device = get_device(cfg)
    data_cfg = cfg["data"]
    checkpoint = load_checkpoint(cfg["unet"]["checkpoint"], map_location=device)
    dataset = SwimsegMaskDataset(data_cfg["prepared_seg_dir"], "test", data_cfg["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["unet"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=device == "cuda",
    )
    model = CloudUNet().to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = _run_unet_epoch(model, loader, nn.BCEWithLogitsLoss(), device)
    print(metrics)
    return metrics


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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cls_cfg["lr"], weight_decay=cls_cfg["weight_decay"])
    best_f1 = -1.0
    output_dir = Path(cfg["project"]["output_dir"]) / "classifier"
    last_checkpoint = output_dir / "last.pt"
    best_checkpoint = output_dir / "best.pt"
    start_epoch = 0
    resume_checkpoint = last_checkpoint if last_checkpoint.exists() else best_checkpoint if best_checkpoint.exists() else None
    if resume_checkpoint is not None:
        checkpoint = load_checkpoint(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_f1 = float(checkpoint.get("best_f1", checkpoint.get("val_metrics", {}).get("f1_macro", -1.0)))
        print(f"Resuming classifier from {resume_checkpoint} at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, cls_cfg["epochs"]):
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
            "optimizer": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "best_f1": max(best_f1, val_metrics["f1_macro"]),
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
