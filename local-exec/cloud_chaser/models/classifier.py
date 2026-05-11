from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision import models

BackboneName = Literal["resnet50", "efficientnet_b0", "densenet121"]


@dataclass(frozen=True)
class BackboneBundle:
    encoder: nn.Module
    features_dim: int


class DenseNetEncoder(nn.Module):
    def __init__(self, model: models.DenseNet) -> None:
        super().__init__()
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.features(x))
        x = self.pool(x)
        return torch.flatten(x, 1)


def build_encoder(backbone: BackboneName = "resnet50", pretrained: bool = True) -> BackboneBundle:
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        features_dim = model.fc.in_features
        model.fc = nn.Identity()
        return BackboneBundle(model, features_dim)
    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        features_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return BackboneBundle(model, features_dim)
    if backbone == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        return BackboneBundle(DenseNetEncoder(model), model.classifier.in_features)
    raise ValueError(f"Unsupported backbone: {backbone}")


class CloudClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: BackboneName = "resnet50",
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        bundle = build_encoder(backbone, pretrained=pretrained)
        self.backbone_name = backbone
        self.encoder = bundle.encoder
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(bundle.features_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def freeze_encoder(self, freeze: bool = True) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
