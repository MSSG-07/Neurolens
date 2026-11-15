"""Small 2D CNN pathway for individual MRI slices."""
import torch
from torch import nn


class Encoder2D(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 32, feature_dim: int = 256, num_classes: int = 2):
        super().__init__()
        widths = (base_channels, base_channels*2, base_channels*4, base_channels*8)
        blocks = []
        current = in_channels
        for width in widths:
            blocks.extend([nn.Conv2d(current, width, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                           nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                           nn.Conv2d(width, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)])
            current = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(nn.Linear(current, feature_dim), nn.ReLU(inplace=True), nn.Linear(feature_dim, feature_dim))
        self.classifier = nn.Linear(current, num_classes)
        self.feature_dim = current

    def forward(self, x):
        if x.ndim != 4: raise ValueError(f"Encoder2D expects (B,C,H,W), got {tuple(x.shape)}")
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        return {"features": features, "pooled": pooled, "embedding": self.projection(pooled), "logits": self.classifier(pooled)}
