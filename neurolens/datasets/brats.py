"""BraTS NIfTI loading and patient-safe paired slice/volume sampling."""
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import normalize_modalities, center_crop_or_pad

try:
    import nibabel as nib
except ImportError:
    nib = None


def convert_brats_labels(labels: np.ndarray) -> np.ndarray:
    """Map standard BraTS {0,1,2,4} labels to contiguous {0,1,2,3}."""
    out = np.nan_to_num(labels).astype(np.uint8, copy=True)
    out[out == 4] = 3
    invalid = ~np.isin(out, (0, 1, 2, 3))
    if invalid.any():
        raise ValueError(f"Unexpected BraTS label values: {np.unique(out[invalid]).tolist()}")
    return out


def split_patients(patient_ids: Sequence[str], val_fraction: float = .2, seed: int = 42):
    """Deterministic patient-level split; each patient appears in exactly one side."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    ids = sorted(set(patient_ids))
    if len(ids) < 2:
        raise ValueError("At least two distinct patients are required for a split")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = min(len(ids) - 1, max(1, round(len(ids) * val_fraction)))
    return ids[n_val:], ids[:n_val]


def discover_patients(root: str, modalities: Sequence[str]):
    """Discover case directories that contain every modality and segmentation."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"BraTS data directory does not exist: {root_path}")
    cases = []
    for case in sorted(p for p in root_path.iterdir() if p.is_dir()):
        segs = list(case.glob("*seg.nii*"))
        mods = [list(case.glob(f"*{mod}*.nii*")) for mod in modalities]
        if len(segs) == 1 and all(len(m) == 1 for m in mods):
            cases.append((case.name, [m[0] for m in mods], segs[0]))
    if not cases:
        raise RuntimeError(f"No complete BraTS cases found under {root_path}; expected modality suffixes {modalities}")
    return cases


class BraTSPairedDataset(Dataset):
    """One random valid slice paired with its originating (cropped) 3D case."""
    def __init__(self, root: str, cases=None, modalities=("t1n", "t1c", "t2w", "t2f"), patch_size=(96,96,96), slice_axis=2):
        if nib is None:
            raise RuntimeError("BraTS loading requires nibabel (pip install nibabel)")
        self.cases = cases or discover_patients(root, modalities)
        self.modalities, self.patch_size, self.slice_axis = tuple(modalities), tuple(patch_size), slice_axis
        if slice_axis not in (0, 1, 2):
            raise ValueError("slice_axis must be 0, 1, or 2")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        patient, modality_paths, seg_path = self.cases[index]
        try:
            # NIfTI files are conventionally (X,Y,Z); convert to (C,D,H,W)=(C,Z,Y,X).
            channels = [np.asarray(nib.load(str(path)).dataobj, dtype=np.float32).transpose(2,1,0) for path in modality_paths]
            label = np.asarray(nib.load(str(seg_path)).dataobj).transpose(2,1,0)
        except Exception as exc:
            raise RuntimeError(f"Failed reading BraTS patient {patient}: {exc}") from exc
        shape = channels[0].shape
        if any(c.shape != shape for c in channels) or label.shape != shape:
            raise ValueError(f"Patient {patient}: modality/label shapes do not match")
        image = center_crop_or_pad(normalize_modalities(np.stack(channels)), self.patch_size)
        label = center_crop_or_pad(convert_brats_labels(label)[None], self.patch_size)[0]
        foreground = np.any(image != 0, axis=0)
        valid = np.flatnonzero(foreground.any(axis=tuple(i for i in range(3) if i != self.slice_axis)))
        if valid.size == 0:
            valid = np.arange(self.patch_size[self.slice_axis])
        slice_idx = int(np.random.choice(valid))
        slice_image = np.take(image, slice_idx, axis=self.slice_axis + 1)
        return {"image2d": torch.from_numpy(slice_image.copy()), "image3d": torch.from_numpy(image.copy()),
                "segmentation": torch.from_numpy(label.astype(np.int64, copy=True)),
                "slice_index": slice_idx, "patient_id": patient, "class_label": torch.tensor(int((label > 0).any()))}
