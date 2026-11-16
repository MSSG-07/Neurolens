"""Volumetric segmentation baselines for the BraTS four-class label space.

These models deliberately implement only 3D segmentation.  The existing BraTS
classification target is binary tumor presence and belongs to NeuroLens' paired
slice/volume pathway; it is not synthesized for segmentation-only baselines.
"""
from __future__ import annotations

import inspect
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int, maximum: int = 8) -> int:
    return next(group for group in range(min(maximum, channels), 0, -1) if channels % group == 0)


class ResidualBlock3D(nn.Module):
    """Two-convolution residual unit with GroupNorm and a shape-matching skip."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.skip = (nn.Identity() if in_channels == out_channels and stride == 1 else
                     nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                                   nn.GroupNorm(_groups(out_channels), out_channels)))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class ResUNet3D(nn.Module):
    """Residual 3D U-Net; returns ``segmentation_logits`` like NeuroLens."""
    def __init__(self, in_channels: int = 4, classes: int = 4, base_channels: int = 16):
        super().__init__()
        if min(in_channels, classes, base_channels) < 1:
            raise ValueError("in_channels, classes and base_channels must be positive")
        widths = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.enc1 = ResidualBlock3D(in_channels, widths[0])
        self.enc2 = ResidualBlock3D(widths[0], widths[1], stride=2)
        self.enc3 = ResidualBlock3D(widths[1], widths[2], stride=2)
        self.enc4 = ResidualBlock3D(widths[2], widths[3], stride=2)
        self.bridge = ResidualBlock3D(widths[3], widths[3] * 2, stride=2)
        self.up4 = nn.ConvTranspose3d(widths[3] * 2, widths[3], 2, stride=2)
        self.dec4 = ResidualBlock3D(widths[3] * 2, widths[3])
        self.up3 = nn.ConvTranspose3d(widths[3], widths[2], 2, stride=2)
        self.dec3 = ResidualBlock3D(widths[2] * 2, widths[2])
        self.up2 = nn.ConvTranspose3d(widths[2], widths[1], 2, stride=2)
        self.dec2 = ResidualBlock3D(widths[1] * 2, widths[1])
        self.up1 = nn.ConvTranspose3d(widths[1], widths[0], 2, stride=2)
        self.dec1 = ResidualBlock3D(widths[0] * 2, widths[0])
        self.segmentation_head = nn.Conv3d(widths[0], classes, 1)

    def forward(self, image3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        if image3d.ndim != 5:
            raise ValueError(f"ResUNet3D expects (B,C,D,H,W), got {tuple(image3d.shape)}")
        a = self.enc1(image3d); b = self.enc2(a); c = self.enc3(b); d = self.enc4(c)
        x = self.dec4(torch.cat((self.up4(self.bridge(d)), d), dim=1))
        x = self.dec3(torch.cat((self.up3(x), c), dim=1))
        x = self.dec2(torch.cat((self.up2(x), b), dim=1))
        x = self.dec1(torch.cat((self.up1(x), a), dim=1))
        logits = self.segmentation_head(x)
        return {"segmentation_logits": logits}


class _ConvPair3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.InstanceNorm3d(out_channels, affine=True), nn.LeakyReLU(0.1, inplace=True),
                         nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                         nn.InstanceNorm3d(out_channels, affine=True), nn.LeakyReLU(0.1, inplace=True))


class UNet3D(nn.Module):
    """Conventional four-level 3D U-Net segmentation reference."""
    def __init__(self, in_channels: int = 4, classes: int = 4, base_channels: int = 16):
        super().__init__()
        if min(in_channels, classes, base_channels) < 1:
            raise ValueError("in_channels, classes and base_channels must be positive")
        c = base_channels
        self.enc1 = _ConvPair3D(in_channels, c); self.enc2 = _ConvPair3D(c, 2*c)
        self.enc3 = _ConvPair3D(2*c, 4*c); self.enc4 = _ConvPair3D(4*c, 8*c)
        self.pool = nn.MaxPool3d(2); self.bridge = _ConvPair3D(8*c, 16*c)
        self.up4 = nn.ConvTranspose3d(16*c, 8*c, 2, 2); self.dec4 = _ConvPair3D(16*c, 8*c)
        self.up3 = nn.ConvTranspose3d(8*c, 4*c, 2, 2); self.dec3 = _ConvPair3D(8*c, 4*c)
        self.up2 = nn.ConvTranspose3d(4*c, 2*c, 2, 2); self.dec2 = _ConvPair3D(4*c, 2*c)
        self.up1 = nn.ConvTranspose3d(2*c, c, 2, 2); self.dec1 = _ConvPair3D(2*c, c)
        self.segmentation_head = nn.Conv3d(c, classes, 1)

    def forward(self, image3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        if image3d.ndim != 5:
            raise ValueError(f"UNet3D expects (B,C,D,H,W), got {tuple(image3d.shape)}")
        a = self.enc1(image3d); b = self.enc2(self.pool(a)); c = self.enc3(self.pool(b)); d = self.enc4(self.pool(c))
        x = self.dec4(torch.cat((self.up4(self.bridge(self.pool(d))), d), 1))
        x = self.dec3(torch.cat((self.up3(x), c), 1)); x = self.dec2(torch.cat((self.up2(x), b), 1))
        x = self.dec1(torch.cat((self.up1(x), a), 1))
        return {"segmentation_logits": self.segmentation_head(x)}


class SwinUNETR3D(nn.Module):
    """Optional MONAI Swin UNETR adapter with a stable NeuroLens-like output key.

    MONAI changed the constructor over releases (notably removal of ``img_size``).
    The adapter inspects the installed signature and passes only supported options.
    MONAI is an optional dependency and is not installed by the base requirements.
    """
    def __init__(self, in_channels: int = 4, classes: int = 4, feature_size: int = 24,
                 use_checkpoint: bool = True, img_size=(64, 64, 64)):
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as exc:
            raise RuntimeError("SwinUNETR requires MONAI; install a compatible MONAI release before enabling it") from exc
        if feature_size % 12:
            raise ValueError("MONAI SwinUNETR feature_size must be a multiple of 12")
        if len(img_size) != 3 or any(size % 32 for size in img_size):
            raise ValueError("SwinUNETR spatial dimensions must each be divisible by 32")
        signature = inspect.signature(SwinUNETR)
        supported = signature.parameters
        kwargs = {"in_channels": in_channels, "out_channels": classes, "feature_size": feature_size,
                  "use_checkpoint": use_checkpoint, "spatial_dims": 3}
        if "img_size" in supported:
            kwargs["img_size"] = tuple(img_size)
        self.network = SwinUNETR(**{k: v for k, v in kwargs.items() if k in supported})

    def forward(self, image3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        if image3d.ndim != 5:
            raise ValueError(f"SwinUNETR3D expects (B,C,D,H,W), got {tuple(image3d.shape)}")
        return {"segmentation_logits": self.network(image3d)}


def build_segmentation_model(name: str, **kwargs) -> nn.Module:
    """Construct one supported volumetric segmentation baseline by registry name."""
    registry = {"resunet3d": ResUNet3D, "unet3d": UNet3D, "swin_unetr": SwinUNETR3D}
    try:
        return registry[name.lower()](**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown segmentation architecture {name!r}; choose {sorted(registry)}") from exc
