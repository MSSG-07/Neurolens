"""MRI normalization and paired spatial operations."""
from typing import Optional
import numpy as np


def normalize_modalities(volume: np.ndarray) -> np.ndarray:
    """Per-modality nonzero z-score normalization; expects (C,D,H,W)."""
    if volume.ndim != 4:
        raise ValueError(f"Expected (C,D,H,W), got {volume.shape}")
    result = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    for channel in range(result.shape[0]):
        image = result[channel]
        mask = image != 0
        if mask.any():
            mean, std = image[mask].mean(), image[mask].std()
            if std > 1e-6:
                image[mask] = (image[mask] - mean) / std
            else:
                image[mask] = 0
        result[channel] = image
    return result


def center_crop_or_pad(array: np.ndarray, size: tuple[int, int, int], fill: float = 0) -> np.ndarray:
    """Center crop or pad the final three dimensions to size."""
    if array.ndim < 3:
        raise ValueError("Input must have at least three dimensions")
    out = np.full((*array.shape[:-3], *size), fill, dtype=array.dtype)
    src_slices, dst_slices = [], []
    for old, new in zip(array.shape[-3:], size):
        take = min(old, new)
        start, dest = max(0, (old - new) // 2), max(0, (new - old) // 2)
        src_slices.append(slice(start, start + take))
        dst_slices.append(slice(dest, dest + take))
    out[(..., *dst_slices)] = array[(..., *src_slices)]
    return out
