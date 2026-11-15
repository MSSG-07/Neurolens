"""Optional independent BRISC 2D classification loader (never paired with BraTS)."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class BRISCDataset(Dataset):
    """Reads class-named image folders; labels are sorted folder names."""
    def __init__(self, root: str, image_size: int = 224):
        self.root, self.image_size = Path(root), image_size
        if not self.root.exists():
            raise FileNotFoundError(f"BRISC directory not found: {self.root}")
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.samples = [(p, i) for i, name in enumerate(self.classes) for p in (self.root/name).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if not self.samples:
            raise RuntimeError("No BRISC images found; expected root/class_name/image files")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB").resize((self.image_size, self.image_size)), dtype=np.float32) / 255.
        return torch.from_numpy(arr.transpose(2,0,1).copy()), torch.tensor(label)
