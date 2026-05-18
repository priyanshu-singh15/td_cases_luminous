"""EfficientNet severity classifier with RGB + edge 4-channel input."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class SeverityNet4Ch(nn.Module):
    def __init__(self, num_classes: int = 3, backbone: str = "efficientnet-b0"):
        super().__init__()
        self.backbone_name = backbone
        self.encoder = EfficientNet.from_name(backbone)
        old_conv = self.encoder._conv_stem
        self.encoder._conv_stem = nn.Conv2d(
            4,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            self.encoder._conv_stem.weight[:, :3] = old_conv.weight
            self.encoder._conv_stem.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
        in_features = self.encoder._fc.in_features
        self.encoder._fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.extract_features(x)
        x = self.encoder._avg_pooling(x)
        x = x.flatten(1)
        return self.head(x)


def load_severity_model(weights_path: Path | None, num_classes: int = 3, device: str = "cpu") -> SeverityNet4Ch:
    model = SeverityNet4Ch(num_classes=num_classes)
    if weights_path and weights_path.exists():
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
