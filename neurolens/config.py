"""Configuration objects shared by datasets, models and training."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    data_root: str = "data/BraTS2023_GLI"
    output_dir: str = "runs/neurolens"
    modalities: Tuple[str, ...] = ("t1n", "t1c", "t2w", "t2f")
    patch_size: Tuple[int, int, int] = (96, 96, 96)
    slice_axis: int = 2
    batch_size: int = 2
    num_workers: int = 2
    accumulation_steps: int = 4
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    embedding_dim: int = 256
    num_classes: int = 4
    classification_classes: int = 2
    base_channels: int = 16
    projection_dim: int = 256
    temperature: float = 0.1
    lambda_classification: float = 1.0
    lambda_segmentation: float = 1.0
    lambda_contrastive: float = 0.1
    lambda_consistency: float = 0.05
    alignment_condition: str = "C"
    seed: int = 42
    val_fraction: float = 0.2
    mixed_precision: bool = True
    device: str = "cuda"
    brats_train_csv: str = ""
    brisc_root: str = ""

    def validate(self) -> None:
        if len(self.patch_size) != 3 or min(self.patch_size) < 16:
            raise ValueError("patch_size must contain three dimensions, each >= 16")
        if any(size % 8 for size in self.patch_size):
            raise ValueError("patch_size dimensions must be divisible by 8 for the 3D decoder")
        if self.batch_size < 1 or self.accumulation_steps < 1:
            raise ValueError("batch_size and accumulation_steps must be positive")
        if self.alignment_condition not in {"A", "B", "C"}:
            raise ValueError("alignment_condition must be A, B, or C")


def load_config(path: str) -> Config:
    """Load YAML when PyYAML exists, otherwise explain the optional dependency."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Reading config.yaml requires PyYAML; pass defaults via Config() otherwise") from exc
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key in ("patch_size", "modalities"):
        if key in values:
            values[key] = tuple(values[key])
    result = Config(**values)
    result.validate()
    return result
