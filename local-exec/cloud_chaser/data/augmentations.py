from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def random_resized_crop(image_size: int, scale: tuple[float, float], ratio: tuple[float, float]):
    """Return RandomResizedCrop for both Albumentations 1.x and 2.x APIs."""
    try:
        return A.RandomResizedCrop(size=(image_size, image_size), scale=scale, ratio=ratio)
    except Exception:
        return A.RandomResizedCrop(height=image_size, width=image_size, scale=scale, ratio=ratio)


def classification_train_transforms(
    image_size: int,
    random_shadow_p: float = 0.25,
    gaussian_blur_p: float = 0.2,
    hflip_p: float = 0.5,
    vflip_p: float = 0.15,
) -> A.Compose:
    """Meteorology-aware classification augmentation.

    Flips preserve cloud texture statistics, while blur and shadows simulate focus,
    haze, occlusion, and illumination changes seen in outdoor sky imagery.
    """
    return A.Compose(
        [
            random_resized_crop(image_size, scale=(0.65, 1.0), ratio=(0.85, 1.2)),
            A.HorizontalFlip(p=hflip_p),
            A.VerticalFlip(p=vflip_p),
            A.RandomShadow(p=random_shadow_p),
            A.GaussianBlur(blur_limit=(3, 5), p=gaussian_blur_p),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.12, hue=0.03, p=0.35),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def eval_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def contrastive_train_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            random_resized_crop(image_size, scale=(0.45, 1.0), ratio=(0.75, 1.33)),
            A.OneOf(
                [
                    A.Rotate(limit=(90, 90), p=1.0),
                    A.Rotate(limit=(180, 180), p=1.0),
                    A.Rotate(limit=(270, 270), p=1.0),
                ],
                p=0.35,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.25),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.04, p=0.6),
            A.GaussNoise(p=0.25),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
