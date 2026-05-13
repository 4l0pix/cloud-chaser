from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloud_chaser.config import get_device
from cloud_chaser.data.augmentations import (
    classification_train_transforms,
    contrastive_train_transforms,
    eval_transforms,
)
from cloud_chaser.data.gcd import GCDDataset, build_gcd_records
from cloud_chaser.data.swimseg import SwimsegMaskDataset, prepare_swimseg_masks
from cloud_chaser.models.classifier import CloudClassifier, ContrastiveCloudEncoder
from cloud_chaser.models.unet import CloudUNet, build_unet
from cloud_chaser.utils.checkpoint import load_checkpoint, save_checkpoint
from cloud_chaser.utils.metrics import classification_metrics
from cloud_chaser.utils.seed import seed_everything


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


def _unet_kwargs(unet_cfg: dict) -> dict[str, object]:
    return {
        "architecture": unet_cfg.get("architecture", "compact"),
        "features": unet_cfg.get("features"),
    }


def _unet_output_dir(cfg: dict, experiment_name: str | None = None) -> Path:
    root = Path(cfg["project"]["output_dir"]) / "unet"
    return root / experiment_name if experiment_name else root


def _run_unet_epoch(
    model: nn.Module,
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


def train_unet_segmenter(cfg: dict, experiment_name: str | None = None, unet_override: dict | None = None) -> Path:
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)
    data_cfg = cfg["data"]
    unet_cfg = {**cfg["unet"], **(unet_override or {})}
    prepare_swimseg_masks(
        root=data_cfg["swimseg_root"],
        output_dir=data_cfg["prepared_seg_dir"],
        val_fraction=data_cfg.get("seg_val_fraction", 0.1),
        test_fraction=data_cfg.get("seg_test_fraction", 0.1),
        seed=cfg["project"]["seed"],
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
    model = build_unet(**_unet_kwargs(unet_cfg)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=unet_cfg["lr"], weight_decay=unet_cfg["weight_decay"])
    output_dir = _unet_output_dir(cfg, experiment_name)
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
            "architecture": unet_cfg.get("architecture", "compact"),
            "features": unet_cfg.get("features"),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "best_miou": max(best_miou, val_metrics["miou"]),
        }
        save_checkpoint(payload, output_dir / "last.pt")
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(payload, output_dir / "best.pt")
    return output_dir / "best.pt"


def evaluate_unet_segmenter(
    cfg: dict,
    checkpoint_path: str | Path | None = None,
    unet_override: dict | None = None,
) -> dict[str, float]:
    device = get_device(cfg)
    data_cfg = cfg["data"]
    checkpoint = load_checkpoint(checkpoint_path or cfg["unet"]["checkpoint"], map_location=device)
    dataset = SwimsegMaskDataset(data_cfg["prepared_seg_dir"], "test", data_cfg["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["unet"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=device == "cuda",
    )
    unet_cfg = {**cfg["unet"], **(unet_override or {})}
    architecture = checkpoint.get("architecture", unet_cfg.get("architecture", "compact"))
    features = checkpoint.get("features", unet_cfg.get("features"))
    model = build_unet(architecture=architecture, features=features).to(device)
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


class ContrastiveGCDDataset(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, image_size: int, classes: list[str], val_fraction: float, seed: int) -> None:
        records, _ = build_gcd_records(root, "train", classes=classes, val_fraction=val_fraction, seed=seed)
        self.paths = [record.path for record in records]
        self.transform = contrastive_train_transforms(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_bgr = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(self.paths[index])
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        q = self.transform(image=image)["image"]
        k = self.transform(image=image)["image"]
        return q, k


def _dequeue_and_enqueue(queue: torch.Tensor, keys: torch.Tensor, ptr: int) -> int:
    batch_size = keys.shape[0]
    queue_size = queue.shape[0]
    if batch_size >= queue_size:
        queue.copy_(keys[-queue_size:])
        return 0
    end = ptr + batch_size
    if end <= queue_size:
        queue[ptr:end] = keys
    else:
        first = queue_size - ptr
        queue[ptr:] = keys[:first]
        queue[: end - queue_size] = keys[first:]
    return end % queue_size


def train_classifier_ssl(cfg: dict) -> Path:
    """MoCo-style CSSL pretraining from the attached ground-cloud classification paper."""
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)
    data_cfg = cfg["data"]
    cls_cfg = cfg["classifier"]
    ssl_cfg = cls_cfg.get("ssl", {})
    output_dir = Path(cfg["project"]["output_dir"]) / "classifier_ssl"
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "last.pt"
    best_checkpoint = output_dir / "best.pt"

    dataset = ContrastiveGCDDataset(
        data_cfg["gcd_root"],
        data_cfg["image_size"],
        data_cfg["classification_classes"],
        data_cfg["val_fraction"],
        cfg["project"]["seed"],
    )
    loader = DataLoader(
        dataset,
        batch_size=ssl_cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=device == "cuda",
        drop_last=True,
    )
    query = ContrastiveCloudEncoder(
        backbone=cls_cfg["backbone"],
        projection_dim=ssl_cfg.get("projection_dim", 128),
        pretrained=True,
    ).to(device)
    key = ContrastiveCloudEncoder(
        backbone=cls_cfg["backbone"],
        projection_dim=ssl_cfg.get("projection_dim", 128),
        pretrained=True,
    ).to(device)
    key.load_state_dict(query.state_dict())
    for param in key.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(
        query.parameters(),
        lr=ssl_cfg.get("lr", 0.03),
        momentum=0.9,
        weight_decay=ssl_cfg.get("weight_decay", 1e-4),
    )
    temperature = ssl_cfg.get("temperature", 0.5)
    momentum = ssl_cfg.get("momentum", 0.999)
    queue_size = ssl_cfg.get("queue_size", 4096)
    projection_dim = ssl_cfg.get("projection_dim", 128)
    queue = F.normalize(torch.randn(queue_size, projection_dim, device=device), dim=1)
    ptr = 0
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    best_loss = float("inf")
    start_epoch = 0
    resume_checkpoint = last_checkpoint if last_checkpoint.exists() else best_checkpoint if best_checkpoint.exists() else None
    if resume_checkpoint is not None and not ssl_cfg.get("force_retrain", False):
        checkpoint = load_checkpoint(resume_checkpoint, map_location=device)
        if "model" in checkpoint:
            query.load_state_dict(checkpoint["model"])
        else:
            query.encoder.load_state_dict(checkpoint["encoder"], strict=False)
            query.projector.load_state_dict(checkpoint["projector"], strict=False)
        key.load_state_dict(query.state_dict())
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError:
                print("Skipping incompatible CSSL optimizer state; continuing with current optimizer.")
        if "queue" in checkpoint:
            queue = checkpoint["queue"].to(device)
        ptr = int(checkpoint.get("queue_ptr", 0))
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_loss = float(checkpoint.get("best_loss", checkpoint.get("loss", best_loss)))
        print(f"Resuming CSSL pretraining from {resume_checkpoint} at epoch {start_epoch + 1}")

    target_epochs = ssl_cfg.get("epochs", 200)
    if start_epoch >= target_epochs:
        print(f"CSSL pretraining already reached target epochs ({target_epochs}).")
        return best_checkpoint if best_checkpoint.exists() else last_checkpoint

    for epoch in range(start_epoch, target_epochs):
        query.train()
        total_loss = 0.0
        for q_images, k_images in tqdm(loader, desc=f"cssl-pretrain-{epoch + 1}"):
            q_images = q_images.to(device, non_blocking=True)
            k_images = k_images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                q = F.normalize(query(q_images), dim=1)
                with torch.no_grad():
                    for q_param, k_param in zip(query.parameters(), key.parameters(), strict=True):
                        k_param.data = k_param.data * momentum + q_param.data * (1.0 - momentum)
                    k = F.normalize(key(k_images), dim=1)
                positive = torch.einsum("nc,nc->n", q, k).unsqueeze(1)
                negative = torch.einsum("nc,kc->nk", q, queue.clone().detach())
                logits = torch.cat([positive, negative], dim=1) / temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ptr = _dequeue_and_enqueue(queue, k.detach(), ptr)
            total_loss += float(loss.detach().cpu())
        avg_loss = total_loss / max(1, len(loader))
        print(f"cssl_epoch={epoch + 1} loss={avg_loss:.4f}")
        payload = {
            "epoch": epoch,
            "backbone": cls_cfg["backbone"],
            "projection_dim": projection_dim,
            "model": query.state_dict(),
            "encoder": query.encoder.state_dict(),
            "projector": query.projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "queue": queue.detach().cpu(),
            "queue_ptr": ptr,
            "loss": avg_loss,
            "best_loss": min(best_loss, avg_loss),
        }
        save_checkpoint(payload, output_dir / "last.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(payload, best_checkpoint)
    return best_checkpoint


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
    ssl_checkpoint = Path(cls_cfg.get("ssl", {}).get("checkpoint", ""))
    if cls_cfg.get("ssl", {}).get("enabled", False) and ssl_checkpoint.exists():
        ssl_payload = load_checkpoint(ssl_checkpoint, map_location=device)
        model.encoder.load_state_dict(ssl_payload["encoder"], strict=False)
        print(f"Loaded CSSL-pretrained encoder from {ssl_checkpoint}")
    criterion = nn.CrossEntropyLoss()
    optimizer_name = str(cls_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cls_cfg["lr"],
            momentum=cls_cfg.get("momentum", 0.9),
            weight_decay=cls_cfg["weight_decay"],
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cls_cfg["lr"],
            weight_decay=cls_cfg["weight_decay"],
        )
    else:
        raise ValueError(f"Unsupported classifier optimizer: {optimizer_name}")
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
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError:
                print("Skipping incompatible classifier optimizer state; continuing with current optimizer.")
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
