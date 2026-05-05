from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from cloud_chaser.models.classifier import BackboneName, build_encoder


class SimCLR(nn.Module):
    def __init__(
        self,
        backbone: BackboneName = "resnet50",
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        bundle = build_encoder(backbone, pretrained=pretrained)
        self.backbone_name = backbone
        self.encoder = bundle.encoder
        self.projector = nn.Sequential(
            nn.Linear(bundle.features_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return F.normalize(self.projector(features), dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -9e15)
    positives = torch.cat(
        [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]
    ).to(z.device)
    return F.cross_entropy(similarity, positives)
