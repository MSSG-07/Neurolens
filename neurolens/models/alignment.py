"""Shared projection space and normalized cross-dimensional embeddings."""
import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim,input_dim), nn.LayerNorm(input_dim), nn.GELU(), nn.Linear(input_dim,output_dim))
    def forward(self, x): return F.normalize(self.layers(x), dim=1)
