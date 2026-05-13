from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

UNetArchitecture = Literal[
    "compact",
    "standard",
    "dilated",
    "dilated_aspp",
    "bicubic",
    "improved",
]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DilatedConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, channels: int, rates: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        branches = []
        for rate in rates:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
        branches.append(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        )
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(channels * len(branches), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DilateBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_aspp: bool) -> None:
        super().__init__()
        self.conv1 = DilatedConv(in_channels, out_channels, dilation=2)
        self.aspp = ASPP(out_channels) if use_aspp else nn.Identity()
        self.conv2 = DilatedConv(out_channels, out_channels, dilation=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.aspp(self.conv1(x)))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSResidualPath(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        return x * torch.sigmoid(self.mlp(avg) + self.mlp(mx))


class SpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            DepthwiseSeparableConv(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attn(x)


class ImprovedCSAM(nn.Module):
    """Practical PyTorch Im-CSAM approximation: channel then spatial attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class SkipProcessor(nn.Module):
    def __init__(self, channels: int, enabled: bool) -> None:
        super().__init__()
        self.block = nn.Sequential(DSResidualPath(channels), ImprovedCSAM(channels)) if enabled else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode == "transpose":
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif mode == "bicubic":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bicubic", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported upsample mode: {mode}")
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class CloudUNet(nn.Module):
    """Configurable U-Net family for binary cloud semantic segmentation."""

    def __init__(
        self,
        architecture: UNetArchitecture = "compact",
        features: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = (64, 128, 256, 512) if architecture == "standard" else (32, 64, 128, 256)
        if len(features) != 4:
            raise ValueError("CloudUNet expects four encoder feature levels.")
        self.architecture = architecture
        self.features = list(features)
        use_dilated = architecture in {"dilated", "dilated_aspp", "bicubic", "improved"}
        use_aspp = architecture in {"dilated_aspp", "bicubic", "improved"}
        use_bicubic = architecture in {"bicubic", "improved"}
        use_attention_skip = architecture == "improved"

        encoder_blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in self.features:
            if use_dilated:
                encoder_blocks.append(DilateBlock(in_channels, out_channels, use_aspp=use_aspp))
            else:
                encoder_blocks.append(DoubleConv(in_channels, out_channels))
            in_channels = out_channels
        self.encoder = nn.ModuleList(encoder_blocks)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(self.features[-1], self.features[-1] * 2)
        self.skip_processors = nn.ModuleList(
            [SkipProcessor(channels, enabled=use_attention_skip) for channels in self.features]
        )

        up_mode = "bicubic" if use_bicubic else "transpose"
        decoder_blocks: list[nn.Module] = []
        current_channels = self.features[-1] * 2
        for skip_channels in reversed(self.features):
            decoder_blocks.append(UpBlock(current_channels, skip_channels, skip_channels, up_mode))
            current_channels = skip_channels
        self.decoder = nn.ModuleList(decoder_blocks)
        self.out = nn.Conv2d(self.features[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        processed = [processor(skip) for processor, skip in zip(self.skip_processors, skips, strict=True)]
        for block, skip in zip(self.decoder, reversed(processed), strict=True):
            x = block(x, skip)
        return self.out(x)


def build_unet(
    architecture: UNetArchitecture = "compact",
    features: list[int] | tuple[int, ...] | None = None,
) -> CloudUNet:
    return CloudUNet(architecture=architecture, features=features)


UNET_EXPERIMENTS: dict[str, dict[str, object]] = {
    "baseline_a_compact": {"architecture": "compact", "features": [32, 64, 128, 256]},
    "baseline_b_medium": {"architecture": "standard", "features": [64, 128, 256, 512]},
    "model_c_dilated": {"architecture": "dilated", "features": [64, 128, 256, 512]},
    "model_d_dilated_aspp": {"architecture": "dilated_aspp", "features": [64, 128, 256, 512]},
    "model_e_bicubic": {"architecture": "bicubic", "features": [64, 128, 256, 512]},
    "model_f_improved": {"architecture": "improved", "features": [64, 128, 256, 512]},
}
