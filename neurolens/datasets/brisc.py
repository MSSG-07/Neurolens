"""Manifest-driven BRISC image classification and mask segmentation datasets."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _truth(value: str) -> bool:
    value = str(value).strip().casefold()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid is_mask value {value!r}; expected a boolean")


def _resolve_path(root: Path, value: str) -> Path:
    """Resolve one manifest path and the two documented task-folder variants."""
    root = Path(root).resolve()
    relative_path = Path(str(value).replace("\\", "/"))
    candidates = [root / relative_path]
    swaps = {
        "classification_task": "classification_task (1)",
        "classification_task (1)": "classification_task",
        "segmentation_task": "segmentation_task (1)",
        "segmentation_task (1)": "segmentation_task",
    }
    # The task folder is the path prefix; retain the complete remaining suffix.
    if relative_path.parts and relative_path.parts[0] in swaps:
        candidates.append(root / Path(swaps[relative_path.parts[0]]).joinpath(*relative_path.parts[1:]))
    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(candidate)
        if root in candidate.parents and candidate.is_file():
            return candidate
    attempted = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(f"BRISC manifest path {value!r} not found; attempted: {attempted}")


def _read_manifest(manifest: str | Path):
    path = Path(manifest)
    if not path.is_file():
        raise FileNotFoundError(f"BRISC manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames or []
    required = {"is_mask", "linked_image", "tumor_label"}
    if not required.issubset(fields):
        raise ValueError(f"BRISC manifest requires {sorted(required)}; found {fields}")
    path_col = next((name for name in ("path", "image_path", "file_path", "filename", "file") if name in fields), None)
    if path_col is None:
        raise ValueError(f"BRISC manifest has no supported file path column; found {fields}")
    return rows, path_col


def _split_rows(rows, split: str | None):
    if split is None:
        return rows
    split_col = next((name for name in ("split", "subset", "partition") if rows and name in rows[0]), None)
    if split_col is None:
        raise ValueError("A split was requested but manifest has no split/subset/partition column")
    aliases = {"val": {"val", "valid", "validation"}, "validation": {"val", "valid", "validation"}}
    wanted = aliases.get(split.casefold(), {split.casefold()})
    selected = [row for row in rows if row[split_col].strip().casefold() in wanted]
    foreign = {row[split_col].strip().casefold() for row in rows} - {"train", "val", "valid", "validation", "test"}
    if foreign:
        raise ValueError(f"Unrecognized official split values: {sorted(foreign)}")
    return selected


class BRISCDataset(Dataset):
    """One BRISC task, selected from the official manifest and split."""
    def __init__(self, root: str | Path, manifest: str | Path | None = None,
                 task: str = "classification", split: str | None = None):
        self.root = Path(root)
        self.manifest = Path(manifest) if manifest else self.root / "manifest.csv"
        if task not in {"classification", "segmentation"}:
            raise ValueError("task must be 'classification' or 'segmentation'")
        if split is None:
            raise ValueError("Specify the official manifest split explicitly (train, val/validation, or test)")
        all_rows, self.path_column = _read_manifest(self.manifest)
        self.class_names = sorted({row["tumor_label"].strip() for row in all_rows
                                   if not _truth(row["is_mask"]) and row["tumor_label"].strip()})
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        rows = all_rows
        rows = _split_rows(rows, split)
        self.task = task
        self.records = []
        by_image = {}
        for row in rows:
            is_mask = _truth(row["is_mask"])
            image_path = _resolve_path(self.root, row[self.path_column])
            if is_mask:
                linked = _resolve_path(self.root, row["linked_image"])
                by_image.setdefault(str(linked), {})["mask"] = (image_path, row)
            else:
                by_image.setdefault(str(image_path), {})["image"] = (image_path, row)
        for key, pair in by_image.items():
            if "image" not in pair:
                if task == "segmentation":
                    raise ValueError(f"Mask row has no linked image row in selected manifest/split: {key}")
                continue
            image_path, image_row = pair["image"]
            if task == "classification":
                label = image_row["tumor_label"].strip()
                if not label:
                    raise ValueError(f"Missing tumor_label for image {image_path}")
                self.records.append((image_path, None, label))
            elif "mask" in pair:
                mask_path, mask_row = pair["mask"]
                if mask_row["linked_image"].strip() == "":
                    raise ValueError(f"Mask row has empty linked_image: {mask_path}")
                self.records.append((image_path, mask_path, image_row["tumor_label"].strip()))
        if not self.records:
            raise ValueError(f"No {task} samples found for split={split!r}")
        for image, mask, _ in self.records:
            if not image.is_file():
                raise FileNotFoundError(f"BRISC image missing: {image}")
            if mask is not None and not mask.is_file():
                raise FileNotFoundError(f"BRISC mask missing: {mask}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, mask_path, label = self.records[index]
        with Image.open(image_path) as im:
            image = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        result = {"image": torch.from_numpy(image.copy()[None]), "tumor_label": label,
                  "class_label": torch.tensor(self.class_to_index[label], dtype=torch.long),
                  "image_path": str(image_path)}
        if mask_path is not None:
            with Image.open(mask_path) as im:
                mask = np.asarray(im)
                if mask.ndim == 3:
                    mask = np.asarray(im.convert("L"))
            if mask.shape != image.shape:
                raise ValueError(f"Image/mask dimensions differ for {image_path}: {image.shape} vs {mask.shape}")
            result["mask"] = torch.from_numpy(mask.copy().astype(np.int64))
            result["mask_path"] = str(mask_path)
        return result


def validate_brisc(root: str | Path, manifest: str | Path | None = None) -> dict:
    """Audit manifest paths, linked pairs, dimensions, labels, duplicates, and task overlap."""
    root = Path(root)
    manifest = Path(manifest) if manifest else root / "manifest.csv"
    rows, path_col = _read_manifest(manifest)
    split_col = next((name for name in ("split", "subset", "partition") if rows and name in rows[0]), None)
    if not split_col:
        raise ValueError("Manifest needs an official split/subset/partition column for audit")
    missing, errors = [], []
    image_records, mask_records = {}, []
    valid_labels = set()
    for number, row in enumerate(rows, start=2):
        try:
            is_mask = _truth(row["is_mask"])
            path = _resolve_path(root, row[path_col])
            if not path.is_file():
                missing.append({"row": number, "path": str(path)})
                continue
            split = row[split_col].strip().casefold()
            if split not in {"train", "val", "valid", "validation", "test"}:
                errors.append(f"row {number}: unrecognized split {split!r}")
            if is_mask:
                linked = _resolve_path(root, row["linked_image"])
                mask_records.append((path, linked, split, row["tumor_label"].strip(), number))
            else:
                label = row["tumor_label"].strip()
                if not label:
                    errors.append(f"row {number}: empty tumor_label")
                valid_labels.add(label)
                image_records[str(path)] = (path, split, label, number)
        except Exception as exc:
            errors.append(f"row {number}: {type(exc).__name__}: {exc}")
    mismatched = []
    overlap = set()
    # Decode/hash each distinct on-disk image at most once, even when several
    # manifest masks point to it. The same cache also supplies pair dimensions.
    image_info = {}
    mask_info = {}
    for path, _, _, _ in image_records.values():
        key = str(path)
        if key in image_info:
            continue
        try:
            with Image.open(path) as im:
                pixels = np.asarray(im)
                descriptor = f"{im.mode}:{im.size}:".encode() + pixels.tobytes()
                image_info[key] = (im.size, hashlib.sha256(descriptor).hexdigest())
        except Exception as exc:
            errors.append(f"image {path}: unreadable: {exc}")
    for mask, linked, split, mask_label, number in mask_records:
        record = image_records.get(str(linked))
        if not record:
            mismatched.append({"row": number, "mask": str(mask), "linked_image": str(linked), "reason": "linked image row absent"})
            continue
        image, image_split, _, _ = record
        if split != image_split:
            mismatched.append({"row": number, "mask": str(mask), "reason": "split mismatch"})
        if mask_label and mask_label != record[2]:
            mismatched.append({"row": number, "mask": str(mask), "reason": f"tumor_label {mask_label!r} != linked image label {record[2]!r}"})
        try:
            image_size = image_info[str(image)][0]
            mask_key = str(mask)
            if mask_key not in mask_info:
                with Image.open(mask) as im:
                    mask_info[mask_key] = im.size
            mask_size = mask_info[mask_key]
            if image_size != mask_size:
                mismatched.append({"row": number, "image": str(image), "mask": str(mask), "reason": f"dimensions {image_size} != {mask_size}"})
            overlap.add(str(image))
        except Exception as exc:
            errors.append(f"row {number}: unreadable image/mask: {exc}")
    hashes = {}
    duplicates = []
    for path, _, _, _ in image_records.values():
        info = image_info.get(str(path))
        if info is None:
            continue
        digest = info[1]
        if digest in hashes:
            duplicates.append({"sha256": digest, "first": hashes[digest], "duplicate": str(path)})
        else:
            hashes[digest] = str(path)
    return {"manifest_rows": len(rows), "image_rows": sum(not _truth(r["is_mask"]) for r in rows),
            "mask_rows": sum(_truth(r["is_mask"]) for r in rows), "split_column": split_col,
            "split_counts": {s: sum(r[split_col].strip().casefold() == s for r in rows) for s in sorted({r[split_col].strip().casefold() for r in rows})},
            "labels": sorted(valid_labels), "missing_files": missing, "mismatched_masks": mismatched,
            "duplicate_image_content": duplicates, "classification_segmentation_image_overlap": len(overlap),
            "errors": errors}
