"""Symmetric in-batch InfoNCE; paired positives share the same row index."""
import torch
from torch import nn
from torch.nn import functional as F


class SymmetricInfoNCE(nn.Module):
    def __init__(self, temperature: float = .1):
        super().__init__()
        if temperature <= 0: raise ValueError("temperature must be positive")
        self.temperature = temperature
    def forward(self, z2d, z3d):
        if z2d.ndim != 2 or z3d.shape != z2d.shape: raise ValueError("Expected matching (B,D) embeddings")
        if z2d.shape[0] < 2: raise ValueError("InfoNCE requires at least two distinct patient pairs per batch")
        a, b = F.normalize(z2d, dim=1), F.normalize(z3d, dim=1)
        logits = a @ b.T / self.temperature
        labels = torch.arange(a.shape[0], device=a.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
