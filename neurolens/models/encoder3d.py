"""Compact 3D CNN encoder with a segmentation decoder."""
import torch
from torch import nn


def conv_block(in_ch, out_ch):
    groups = min(8, out_ch)
    while out_ch % groups:
        groups -= 1
    return nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(groups, out_ch), nn.GELU(),
                         nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(groups, out_ch), nn.GELU())


class Encoder3D(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 16, feature_dim: int = 256, num_classes: int = 4):
        super().__init__()
        c = base_channels
        self.enc1, self.enc2 = conv_block(in_channels,c), conv_block(c,c*2)
        self.enc3, self.enc4 = conv_block(c*2,c*4), conv_block(c*4,c*8)
        self.pool, self.pool3 = nn.MaxPool3d(2), nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Sequential(nn.Linear(c*8, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim))
        self.up3, self.dec3 = nn.ConvTranspose3d(c*8,c*4,2,2), conv_block(c*8,c*4)
        self.up2, self.dec2 = nn.ConvTranspose3d(c*4,c*2,2,2), conv_block(c*4,c*2)
        self.up1, self.dec1 = nn.ConvTranspose3d(c*2,c,2,2), conv_block(c*2,c)
        self.segmentation_head = nn.Conv3d(c,num_classes,1)

    def forward(self, x):
        if x.ndim != 5: raise ValueError(f"Encoder3D expects (B,C,D,H,W), got {tuple(x.shape)}")
        a = self.enc1(x); b = self.enc2(self.pool(a)); c = self.enc3(self.pool(b)); d = self.enc4(self.pool(c))
        z = self.pool3(d).flatten(1)
        u = self.dec3(torch.cat((self.up3(d), c), 1))
        u = self.dec2(torch.cat((self.up2(u), b), 1))
        u = self.dec1(torch.cat((self.up1(u), a), 1))
        return {"features": d, "pooled": z, "embedding": self.projection(z), "segmentation_logits": self.segmentation_head(u)}
