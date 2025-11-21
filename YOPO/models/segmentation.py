"""Lightweight segmentation network for safety mask prediction."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class UNetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        bottleneck = self.pool3(x3)
        return x1, x2, x3, bottleneck


class UNetDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv2 = DoubleConv(64, 32)
        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv3 = DoubleConv(32, 16)
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, bottleneck: torch.Tensor) -> torch.Tensor:
        d1 = self.up1(bottleneck)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.conv1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv2(d2)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.conv3(d3)
        return self.out_conv(d3)


class LightweightUNet(nn.Module):
    """Simplified U-Net for binary safety segmentation."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, bottleneck = self.encoder(x)
        logits = self.decoder(x1, x2, x3, bottleneck)
        return logits

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target)
