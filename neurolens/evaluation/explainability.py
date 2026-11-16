"""Gradient-based activation maps for convolutional 2D/3D model pathways.

Grad-CAM visualizes gradient-weighted activations. It is not a causal
explanation. For transformers use a validated attention attribution method;
this module does not mislabel attention maps as Grad-CAM.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F


class GradCAM:
    """Compute Grad-CAM for a chosen convolutional layer and scalar model target.

    ``forward_fn(model, inputs)`` returns logits or a mapping containing a
    ``logits_key`` tensor. A target class is selected per batch item. For
    segmentation, ``target_mask`` can limit the logit sum to a spatial region.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module,
                 forward_fn: Callable[[nn.Module, torch.Tensor], Union[torch.Tensor, dict]],
                 logits_key: Optional[str] = None):
        self.model, self.target_layer, self.forward_fn = model, target_layer, forward_fn
        self.logits_key = logits_key

    def __call__(self, inputs: torch.Tensor, target_class: Union[int, torch.Tensor],
                 target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        activations, gradients = {}, {}
        def capture(_module, _args, output):
            if not torch.is_tensor(output) or output.ndim not in (4, 5):
                raise ValueError("Grad-CAM target layer must emit a 2D or 3D convolutional feature map")
            activations["value"] = output
            output.register_hook(lambda grad: gradients.__setitem__("value", grad))
        handle = self.target_layer.register_forward_hook(capture)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                x = inputs.detach().requires_grad_(True)
                output = self.forward_fn(self.model, x)
                if isinstance(output, dict):
                    if self.logits_key is None or self.logits_key not in output:
                        raise KeyError(f"Specify logits_key present in model output; available={list(output)}")
                    output = output[self.logits_key]
                if output.ndim < 2:
                    raise ValueError(f"Target output needs batch and class dimensions, got {tuple(output.shape)}")
                classes = torch.as_tensor(target_class, device=output.device, dtype=torch.long)
                if classes.ndim == 0: classes = classes.expand(output.shape[0])
                if classes.shape != (output.shape[0],):
                    raise ValueError("target_class must be a scalar or one class index per batch item")
                class_indices = classes.view(-1, 1, *([1] * (output.ndim - 2)))
                if output.ndim > 2:
                    class_indices = class_indices.expand(-1, 1, *output.shape[2:])
                selected = output.gather(1, class_indices)
                if target_mask is not None:
                    mask = target_mask.to(device=output.device, dtype=selected.dtype)
                    if mask.ndim == output.ndim - 1: mask = mask.unsqueeze(1)
                    selected = selected * F.interpolate(mask, size=output.shape[2:], mode="nearest")
                score = selected.reshape(output.shape[0], -1).sum(dim=1).sum()
                if "value" not in activations:
                    raise RuntimeError("Target layer was not reached by forward_fn")
                grads = torch.autograd.grad(score, activations["value"], retain_graph=False)[0]
                dims = tuple(range(2, grads.ndim))
                weights = grads.mean(dim=dims, keepdim=True)
                cam = torch.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=inputs.shape[2:], mode="trilinear" if inputs.ndim == 5 else "bilinear", align_corners=False)
                minimum = cam.amin(dim=tuple(range(2, cam.ndim)), keepdim=True)
                maximum = cam.amax(dim=tuple(range(2, cam.ndim)), keepdim=True)
                cam = (cam - minimum) / (maximum - minimum).clamp_min(torch.finfo(cam.dtype).eps)
                if not torch.isfinite(cam).all(): raise FloatingPointError("Grad-CAM produced non-finite values")
                return cam.detach()
        finally:
            handle.remove()
            self.model.train(was_training)


def neurolens_2d_classification_cam(model: nn.Module) -> GradCAM:
    """Grad-CAM on NeuroLens' 2D encoder for its binary classification logits."""
    layer = model.encoder2d.features[-1]
    return GradCAM(model, layer, lambda m, x: m.encoder2d(x), logits_key="logits")


def neurolens_3d_segmentation_cam(model: nn.Module) -> GradCAM:
    """Grad-CAM on NeuroLens' 3D encoder for segmentation logits."""
    layer = model.encoder3d.enc4[-1]
    return GradCAM(model, layer, lambda m, x: m.encoder3d(x), logits_key="segmentation_logits")


def neurolens_2d_embedding_cam(model: nn.Module) -> GradCAM:
    """Explain one projected 2D embedding coordinate at the 2D feature layer."""
    layer = model.encoder2d.features[-1]
    return GradCAM(model, layer,
                   lambda m, x: {"embedding":m.project2d(m.encoder2d(x)["embedding"])},
                   logits_key="embedding")


def neurolens_3d_embedding_cam(model: nn.Module) -> GradCAM:
    """Explain one projected 3D embedding coordinate at the 3D feature layer."""
    layer = model.encoder3d.enc4[-1]
    return GradCAM(model, layer,
                   lambda m, x: {"embedding":m.project3d(m.encoder3d(x)["embedding"])},
                   logits_key="embedding")


def activation_overlap(cam: torch.Tensor, tumor_mask: torch.Tensor, threshold: float = 0.5) -> dict:
    """Summarize thresholded activation/tumor overlap per case, without causal claims."""
    if not 0 < threshold < 1: raise ValueError("threshold must be between 0 and 1")
    if cam.shape[0] != tumor_mask.shape[0]: raise ValueError("CAM and mask batch sizes differ")
    activation = cam.squeeze(1) >= threshold
    tumor = tumor_mask.to(device=activation.device) > 0
    if activation.shape != tumor.shape: raise ValueError("CAM and tumor mask spatial shapes must match")
    dims = tuple(range(1, activation.ndim))
    intersection = (activation & tumor).sum(dim=dims).float()
    union = (activation | tumor).sum(dim=dims).float()
    dice_den = activation.sum(dim=dims) + tumor.sum(dim=dims)
    total = torch.tensor(float(activation[0].numel()), device=activation.device)
    return {"tumor_activation_ratio": ((activation & tumor).sum(dim=dims) / tumor.sum(dim=dims).clamp_min(1)).cpu().tolist(),
            "background_activation_ratio": ((activation & ~tumor).sum(dim=dims) / (~tumor).sum(dim=dims).clamp_min(1)).cpu().tolist(),
            "dice": torch.where(dice_den > 0, 2 * intersection / dice_den, torch.ones_like(intersection)).cpu().tolist(),
            "iou": torch.where(union > 0, intersection / union, torch.ones_like(union)).cpu().tolist(),
            "threshold": threshold, "activation_fraction": (activation.sum(dim=dims) / total).cpu().tolist()}
