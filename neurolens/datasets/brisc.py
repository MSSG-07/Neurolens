"""Manifest-driven BRISC image classification and mask segmentation datasets."""
from __future__ import annotations

import csv
import hashlib
import json
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


class BRISCPathResolver:
    """Resolve manifest paths safely across the two known task-folder variants."""
    _variants = {
        "classification_task": "classification_task (1)",
        "classification_task (1)": "classification_task",
        "segmentation_task": "segmentation_task (1)",
        "segmentation_task (1)": "segmentation_task",
    }

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._hash_cache = {}
        self.resolution_counts = {"canonical": 0, "alternate": 0}
        self.last_call = None

    def resolve(self, value: str | Path) -> Path:
        original = str(value)
        self.last_call = {"value": original, "type": type(value).__name__}
        normalized = Path(original.replace("\\", "/"))
        if normalized.is_absolute() or any(part == ".." for part in normalized.parts):
            raise ValueError(f"Unsafe BRISC manifest path {original!r}: paths must stay within {self.root}")
        candidates = [self.root / normalized]
        if normalized.parts and normalized.parts[0] in self._variants:
            alternative = Path(self._variants[normalized.parts[0]]).joinpath(*normalized.parts[1:])
            candidates.append(self.root / alternative)
        checked = [candidate.resolve() for candidate in candidates]
        checked = [path for path in checked if self.root in path.parents]
        existing = [path for path in checked if path.is_file()]
        if len(existing) == 1:
            self.resolution_counts["canonical" if existing[0] == checked[0] else "alternate"] += 1
            return existing[0]
        if len(existing) > 1:
            hashes = {path: self._sha256(path) for path in existing}
            if len(set(hashes.values())) == 1:
                # Both exported copies are byte-identical; preserve manifest preference.
                self.resolution_counts["canonical"] += 1
                return existing[0]
            raise RuntimeError(
                f"Ambiguous BRISC manifest path {original!r}: multiple different files exist: "
                + ", ".join(str(path) for path in existing)
            )
        attempted = ", ".join(str(path) for path in checked)
        raise FileNotFoundError(f"BRISC manifest path {original!r} not found; attempted: {attempted}")

    def _sha256(self, path: Path) -> str:
        path = Path(path)
        if path not in self._hash_cache:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._hash_cache[path] = digest.hexdigest()
        return self._hash_cache[path]


def _resolve_path(root: str | Path, value: str | Path) -> Path:
    """Compatibility wrapper; production callers reuse a BRISCPathResolver."""
    return BRISCPathResolver(root).resolve(value)


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
    path_col = next((name for name in ("relative_path", "path", "image_path", "file_path", "file") if name in fields), None)
    if path_col is None and "filename" not in fields:
        raise ValueError(f"BRISC manifest requires relative_path or filename; found {fields}")
    return rows, path_col


def _resolve_row_path(resolver: BRISCPathResolver, row: dict, path_col: str | None) -> Path:
    relative = str(row.get(path_col, "") or "").strip() if path_col else ""
    filename = str(row.get("filename", "") or "").strip()
    if relative:
        try:
            return resolver.resolve(relative)
        except FileNotFoundError as relative_error:
            if filename and filename != relative:
                try:
                    return resolver.resolve(filename)
                except FileNotFoundError as filename_error:
                    raise FileNotFoundError(f"Neither relative_path nor filename resolved. {relative_error}; {filename_error}") from filename_error
            raise
    if filename:
        return resolver.resolve(filename)
    raise ValueError("Manifest row has neither a usable relative_path nor filename")


def _json_manifest_rows(value):
    """Yield JSON objects that look like manifest rows, without assuming a schema."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if any(key in item for key in ("relative_path", "filename", "is_mask")):
                yield item
            else:
                stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _task_for_path(path: Path, root: Path) -> str | None:
    parts = Path(path).relative_to(root).parts
    if parts and parts[0].startswith("classification_task"):
        return "classification"
    if parts and parts[0].startswith("segmentation_task"):
        return "segmentation"
    return None


def _trace_fields(row: dict) -> dict:
    relative = str(row.get("relative_path", "") or "")
    return {"relative_path": row.get("relative_path"), "filename": row.get("filename"),
            "task": row.get("task") or _path_task_name(relative),
            "split": row.get("split") or row.get("subset") or row.get("partition"),
            "linked_image": row.get("linked_image"), "is_mask": row.get("is_mask")}


def _path_task_name(value: str) -> str | None:
    parts = Path(str(value).replace("\\", "/")).parts
    return parts[0] if parts else None


def _matches_trace(row: dict, trace_filename: str | None) -> bool:
    if not trace_filename:
        return False
    target = trace_filename.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return any(str(row.get(key, "") or "").replace("\\", "/").rsplit("/", 1)[-1].casefold() == target
               for key in ("relative_path", "filename", "linked_image"))


def _manifest_image_index(rows, paths, split_col: str, root: Path):
    """Index resolved non-mask records by verified split/task/basename metadata."""
    index = {}
    for row_number, row in enumerate(rows):
        if _truth(row["is_mask"]) or row_number not in paths:
            continue
        path = paths[row_number]
        task = _task_for_path(path, root)
        if task is None:
            continue
        split = row[split_col].strip().casefold()
        names = {path.name.casefold(), str(row.get("filename", "")).strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()}
        for name in names - {""}:
            key = (split, task, name)
            index.setdefault(key, {})[str(path)] = path
    return index


def _resolve_linked_image(value: str, mask_path: Path, split: str, resolver: BRISCPathResolver, image_index,
                          return_source: bool = False):
    """Resolve full linked paths directly; resolve basenames only via manifest rows."""
    normalized = Path(str(value).replace("\\", "/"))
    if len(normalized.parts) > 1:
        path = resolver.resolve(value)
        return (path, value) if return_source else path
    task = _task_for_path(mask_path, resolver.root)
    candidates = image_index.get((split.casefold(), task, normalized.name.casefold()), {})
    if len(candidates) == 1:
        # The full relative path comes from an image manifest row, never a folder guess.
        path = next(iter(candidates.values()))
        source = path.relative_to(resolver.root).as_posix()
        resolved = resolver.resolve(source)
        return (resolved, source) if return_source else resolved
    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous linked_image basename {str(value)!r}: " + ", ".join(candidates))
    raise FileNotFoundError(
        f"BRISC linked_image basename {str(value)!r} has no matching non-mask manifest row "
        f"for task={task!r}, split={split!r}"
    )


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
        self.resolver = BRISCPathResolver(self.root)
        # Preserve the dataset's named classes explicitly; do not derive meaning
        # from filesystem or alphabetical ordering.
        self.class_names = ("glioma", "meningioma", "pituitary", "no_tumor")
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        rows = _split_rows(all_rows, split)
        split_col = next(name for name in ("split", "subset", "partition") if rows and name in rows[0])
        resolved_rows = [(row, _resolve_row_path(self.resolver, row, self.path_column)) for row in rows]
        image_index = _manifest_image_index(
            rows, {number: path for number, (_, path) in enumerate(resolved_rows)}, split_col, self.resolver.root
        )
        self.task = task
        self.records = []
        by_image = {}
        for row, image_path in resolved_rows:
            is_mask = _truth(row["is_mask"])
            first_component = image_path.relative_to(self.resolver.root).parts[:1]
            task_match = bool(first_component and first_component[0].startswith(f"{task}_task"))
            if not task_match:
                continue
            if is_mask:
                linked = _resolve_linked_image(row["linked_image"], image_path, row[split_col], self.resolver, image_index)
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
                label = image_row["tumor_label"].strip().casefold()
                if not label:
                    raise ValueError(f"Missing tumor_label for image {image_path}")
                if label not in self.class_to_index:
                    raise ValueError(f"Unknown BRISC tumor_label {label!r} for {image_path}; expected {self.class_names}")
                self.records.append((image_path, None, label))
            elif "mask" in pair:
                mask_path, mask_row = pair["mask"]
                if mask_row["linked_image"].strip() == "":
                    raise ValueError(f"Mask row has empty linked_image: {mask_path}")
                label = image_row["tumor_label"].strip().casefold()
                if label not in self.class_to_index:
                    raise ValueError(f"Unknown BRISC tumor_label {label!r} for {image_path}; expected {self.class_names}")
                self.records.append((image_path, mask_path, label))
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
                if mask.ndim != 2:
                    raise ValueError(f"Segmentation mask must be a single-channel label image, got mode={im.mode}, shape={mask.shape}: {mask_path}")
            if mask.shape != image.shape:
                raise ValueError(f"Image/mask dimensions differ for {image_path}: {image.shape} vs {mask.shape}")
            result["mask"] = torch.from_numpy(mask.copy().astype(np.int64))
            result["mask_path"] = str(mask_path)
        return result


def validate_brisc(root: str | Path, manifest: str | Path | None = None,
                   trace_filename: str | None = None) -> dict:
    """Audit manifest paths, linked pairs, dimensions, labels, duplicates, and task overlap."""
    root = Path(root).resolve()
    manifest = Path(manifest) if manifest else root / "manifest.csv"
    rows, path_col = _read_manifest(manifest)
    resolver = BRISCPathResolver(root)
    split_col = next((name for name in ("split", "subset", "partition") if rows and name in rows[0]), None)
    if not split_col:
        raise ValueError("Manifest needs an official split/subset/partition column for audit")
    missing, ambiguous, corrupt, errors = [], [], [], []
    trace_rows = []
    row_paths = {}
    row_resolver_calls = {}
    for row_number, row in enumerate(rows, start=2):
        try:
            row_paths[row_number - 2] = _resolve_row_path(resolver, row, path_col)
            row_resolver_calls[row_number - 2] = resolver.last_call
        except RuntimeError as exc:
            row_resolver_calls[row_number - 2] = resolver.last_call
            item = {"row": row_number, "error": str(exc), "manifest_fields": _trace_fields(row),
                    "resolver_value": resolver.last_call["value"] if resolver.last_call else None,
                    "resolver_value_type": resolver.last_call["type"] if resolver.last_call else None,
                    "resolution_role": "row_path"}
            ambiguous.append(item)
            if _matches_trace(row, trace_filename):
                trace_rows.append(item)
        except FileNotFoundError as exc:
            row_resolver_calls[row_number - 2] = resolver.last_call
            item = {"row": row_number, "error": str(exc), "manifest_fields": _trace_fields(row),
                    "resolver_value": resolver.last_call["value"] if resolver.last_call else None,
                    "resolver_value_type": resolver.last_call["type"] if resolver.last_call else None,
                    "resolution_role": "row_path"}
            missing.append(item)
            if _matches_trace(row, trace_filename):
                trace_rows.append(item)
        except Exception as exc:
            row_resolver_calls[row_number - 2] = resolver.last_call
            errors.append(f"row {row_number}: {type(exc).__name__}: {exc}")
    image_index = _manifest_image_index(rows, row_paths, split_col, root)
    image_records, mask_records = {}, []
    labels_by_task = {"classification": set(), "segmentation": set()}
    task_split_counts = {}
    mask_split_counts = {}
    task_label_counts = {}
    planes = set()
    duplicate_references = []
    seen_paths = {}
    for number, row in enumerate(rows, start=2):
        row_index = number - 2
        if row_index not in row_paths:
            continue
        path = row_paths[row_index]
        trace_item = None
        try:
            is_mask = _truth(row["is_mask"])
            row_resolver_call = row_resolver_calls.get(row_index)
            split = row[split_col].strip().casefold()
            if split not in {"train", "val", "valid", "validation", "test"}:
                errors.append(f"row {number}: unrecognized split {split!r}")
            relative_parts = path.relative_to(root).parts
            task = "classification" if relative_parts and relative_parts[0].startswith("classification_task") else "segmentation" if relative_parts and relative_parts[0].startswith("segmentation_task") else "unknown"
            if task == "unknown":
                errors.append(f"row {number}: path is outside a recognized BRISC task: {path}")
            if not is_mask and task != "unknown":
                count_key = f"{task}_{split}_images"
                task_split_counts[count_key] = task_split_counts.get(count_key, 0) + 1
            elif is_mask and task == "segmentation":
                count_key = f"segmentation_{split}_masks"
                mask_split_counts[count_key] = mask_split_counts.get(count_key, 0) + 1
            ref_key = str(path)
            if ref_key in seen_paths:
                first_row, first_split = seen_paths[ref_key]
                duplicate_references.append({"path": ref_key, "first_row": first_row, "first_split": first_split,
                                             "duplicate_row": number, "duplicate_split": split,
                                             "cross_split": first_split != split})
            else:
                seen_paths[ref_key] = (number, split)
            label = row["tumor_label"].strip().casefold()
            if not label:
                errors.append(f"row {number}: empty tumor_label")
            elif label not in {"glioma", "meningioma", "pituitary", "no_tumor"}:
                errors.append(f"row {number}: unsupported tumor_label {label!r}")
            if task in labels_by_task:
                labels_by_task[task].add(label)
                label_key = f"{task}_{split}_{label}"
                task_label_counts[label_key] = task_label_counts.get(label_key, 0) + (0 if is_mask else 1)
            for column in ("plane", "plane_label"):
                if row.get(column, "").strip():
                    plane = row[column].strip().casefold()
                    planes.add(plane)
                    if plane not in {"ax", "axial", "cor", "coronal", "sag", "sagittal"}:
                        errors.append(f"row {number}: unsupported {column} {plane!r}")
            if is_mask:
                linked, linked_source = _resolve_linked_image(row["linked_image"], path, split, resolver,
                                                                image_index, return_source=True)
                mask_records.append((path, linked, split, label, number))
                linked_trace = {"raw_linked_image": row["linked_image"],
                                "raw_linked_image_type": type(row["linked_image"]).__name__,
                                "resolver_value": linked_source, "resolver_value_type": type(linked_source).__name__,
                                "resolved_path": str(linked), "resolution_role": "linked_image"}
            else:
                image_records[str(path)] = (path, split, label, number)
                linked_trace = None
            if _matches_trace(row, trace_filename):
                trace_item = {"row": number, "manifest_fields": _trace_fields(row),
                              "resolver_value": row_resolver_call["value"] if row_resolver_call else None,
                              "resolver_value_type": row_resolver_call["type"] if row_resolver_call else None,
                              "resolved_path": str(path), "resolution_role": "row_path"}
                if linked_trace:
                    trace_item["linked_image_resolution"] = linked_trace
                trace_rows.append(trace_item)
        except RuntimeError as exc:
            if "Ambiguous BRISC manifest path" in str(exc):
                linked_is_basename = is_mask and len(Path(str(row.get("linked_image", "")).replace("\\", "/")).parts) == 1
                item = {"row": number, "error": str(exc), "manifest_fields": _trace_fields(row),
                        "resolver_value": None if linked_is_basename else (resolver.last_call["value"] if resolver.last_call else None),
                        "resolver_value_type": None if linked_is_basename else (resolver.last_call["type"] if resolver.last_call else None),
                        "manifest_lookup_value": row.get("linked_image") if linked_is_basename else None,
                        "manifest_lookup_value_type": type(row.get("linked_image")).__name__ if linked_is_basename else None,
                        "resolution_role": "linked_image" if is_mask else "row_path"}
                ambiguous.append(item)
                if _matches_trace(row, trace_filename):
                    trace_rows.append(item)
            else:
                errors.append(f"row {number}: {type(exc).__name__}: {exc}")
        except FileNotFoundError as exc:
            linked_is_basename = is_mask and len(Path(str(row.get("linked_image", "")).replace("\\", "/")).parts) == 1
            item = {"row": number, "error": str(exc), "manifest_fields": _trace_fields(row),
                    "resolver_value": None if linked_is_basename else (resolver.last_call["value"] if resolver.last_call else None),
                    "resolver_value_type": None if linked_is_basename else (resolver.last_call["type"] if resolver.last_call else None),
                    "manifest_lookup_value": row.get("linked_image") if linked_is_basename else None,
                    "manifest_lookup_value_type": type(row.get("linked_image")).__name__ if linked_is_basename else None,
                    "resolution_role": "linked_image" if is_mask else "row_path"}
            missing.append(item)
            if _matches_trace(row, trace_filename):
                trace_rows.append(item)
        except Exception as exc:
            errors.append(f"row {number}: {type(exc).__name__}: {exc}")
    mismatched = []
    overlap = set()
    # Decode/hash each distinct on-disk image at most once, even when several
    # manifest masks point to it. The same cache also supplies pair dimensions.
    image_info = {}
    mask_info = {}
    image_modes = set()
    mask_values = set()
    for path, _, _, _ in image_records.values():
        key = str(path)
        if key in image_info:
            continue
        try:
            with Image.open(path) as im:
                pixels = np.asarray(im)
                descriptor = f"{im.mode}:{im.size}:".encode() + pixels.tobytes()
                image_info[key] = (im.size, hashlib.sha256(descriptor).hexdigest())
                image_modes.add(im.mode)
        except Exception as exc:
            corrupt.append({"path": str(path), "error": str(exc)})
    for mask, _, _, _, number in mask_records:
        mask_key = str(mask)
        if mask_key in mask_info:
            continue
        try:
            with Image.open(mask) as im:
                pixels = np.asarray(im)
                if pixels.ndim != 2:
                    raise ValueError(f"Segmentation mask must be single-channel, got mode={im.mode}, shape={pixels.shape}")
                descriptor = f"{im.mode}:{im.size}:".encode() + pixels.tobytes()
                values = np.unique(pixels).tolist()
                mask_values.update(int(value) for value in values)
                mask_info[mask_key] = (im.size, hashlib.sha256(descriptor).hexdigest(), values)
        except Exception as exc:
            corrupt.append({"row": number, "path": str(mask), "error": str(exc)})
    valid_pairs = 0
    for mask, linked, split, mask_label, number in mask_records:
        record = image_records.get(str(linked))
        if not record:
            mismatched.append({"row": number, "mask": str(mask), "linked_image": str(linked), "reason": "linked image row absent"})
            continue
        image, image_split, _, _ = record
        pair_valid = True
        if split != image_split:
            mismatched.append({"row": number, "mask": str(mask), "reason": "split mismatch"})
            pair_valid = False
        if mask_label and mask_label != record[2]:
            mismatched.append({"row": number, "mask": str(mask), "reason": f"tumor_label {mask_label!r} != linked image label {record[2]!r}"})
            pair_valid = False
        if mask.stem != image.stem:
            mismatched.append({"row": number, "mask": str(mask), "image": str(image), "reason": "image-mask basename mismatch"})
            pair_valid = False
        try:
            image_size = image_info[str(image)][0]
            mask_key = str(mask)
            if mask_key not in mask_info:
                raise ValueError(f"Mask file could not be decoded: {mask}")
            mask_size = mask_info[mask_key][0]
            if image_size != mask_size:
                mismatched.append({"row": number, "image": str(image), "mask": str(mask), "reason": f"dimensions {image_size} != {mask_size}"})
                pair_valid = False
            overlap.add(str(image))
            if pair_valid:
                valid_pairs += 1
        except Exception as exc:
            corrupt.append({"row": number, "image": str(image), "mask": str(mask), "error": str(exc)})
    hashes = {}
    duplicates = []
    for path, split_value, _, _ in image_records.values():
        info = image_info.get(str(path))
        if info is None:
            continue
        digest = info[1]
        if digest in hashes:
            first_path, first_split = hashes[digest]
            duplicates.append({"sha256": digest, "first": first_path, "first_split": first_split,
                               "duplicate": str(path), "duplicate_split": split_value,
                               "cross_split": first_split != split_value})
        else:
            hashes[digest] = (str(path), split_value)
    mask_hashes = {}
    duplicate_mask_content = []
    for mask, _, split_value, _, _ in mask_records:
        info = mask_info.get(str(mask))
        if info is None:
            continue
        digest = info[1]
        if digest in mask_hashes and mask_hashes[digest][0] != str(mask):
            first_path, first_split = mask_hashes[digest]
            duplicate_mask_content.append({"sha256": digest, "first": first_path, "first_split": first_split,
                                           "duplicate": str(mask), "duplicate_split": split_value,
                                           "cross_split": first_split != split_value})
        else:
            mask_hashes[digest] = (str(mask), split_value)
    json_path = manifest.with_name("manifest.json")
    json_metadata = {"exists": json_path.is_file()}
    if not json_path.is_file():
        errors.append(f"Expected BRISC JSON manifest is missing: {json_path}")
    else:
        try:
            json_data = json.loads(json_path.read_text(encoding="utf-8"))
            json_rows = list(_json_manifest_rows(json_data))
            json_path_errors = []
            json_path_map = {}
            json_split_col = next((name for name in ("split", "subset", "partition")
                                   if json_rows and name in json_rows[0]), None)
            for number, row in enumerate(json_rows, start=1):
                try:
                    json_path_map[number - 1] = _resolve_row_path(
                        resolver, row, "relative_path" if "relative_path" in row else None
                    )
                except Exception as exc:
                    json_path_errors.append({"row": number, "error": f"{type(exc).__name__}: {exc}"})
            if json_split_col:
                json_image_index = _manifest_image_index(json_rows, json_path_map, json_split_col, root)
            else:
                json_image_index = _manifest_image_index(
                    json_rows, json_path_map, "", root
                ) if all("" in row for row in json_rows) else {}
            for number, row in enumerate(json_rows, start=1):
                if not row.get("is_mask") or number - 1 not in json_path_map:
                    continue
                split_value = row.get(json_split_col, "") if json_split_col else ""
                try:
                    _resolve_linked_image(row.get("linked_image", ""), json_path_map[number - 1],
                                          split_value, resolver, json_image_index)
                except Exception as exc:
                    json_path_errors.append({"row": number, "error": f"{type(exc).__name__}: {exc}",
                                              "linked_image": row.get("linked_image")})
            json_metadata.update({"type": type(json_data).__name__, "keys": list(json_data)[:30] if isinstance(json_data, dict) else None,
                                 "manifest_like_entries": len(json_rows), "path_errors": json_path_errors})
            if json_path_errors:
                errors.append(f"manifest.json contains {len(json_path_errors)} unresolved/invalid path entries")
        except Exception as exc:
            json_metadata["error"] = f"{type(exc).__name__}: {exc}"
    checksums = {}
    for file in (manifest, json_path):
        candidates = (file.with_suffix(file.suffix + ".sha256"), file.with_suffix(".sha256"))
        checksum_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if checksum_path:
            try:
                tokens = checksum_path.read_text(encoding="utf-8").split()
                if not tokens:
                    raise ValueError("empty checksum file")
                expected = tokens[0]
                actual = hashlib.sha256(file.read_bytes()).hexdigest()
                matches = expected.casefold() == actual.casefold()
                checksums[file.name] = {"path": str(checksum_path), "expected": expected, "actual": actual, "matches": matches}
                if not matches:
                    errors.append(f"SHA256 mismatch for {file}: expected {expected}, got {actual}")
            except Exception as exc:
                errors.append(f"Invalid checksum file {checksum_path}: {type(exc).__name__}: {exc}")
    task_hashes = {"classification": set(), "segmentation": set()}
    for path_text, (path, _, _, _) in image_records.items():
        info = image_info.get(path_text)
        if info:
            parts = path.relative_to(root).parts
            task = "classification" if parts and parts[0].startswith("classification_task") else "segmentation" if parts and parts[0].startswith("segmentation_task") else None
            if task:
                task_hashes[task].add(info[1])
    classification_paths = {path for path, record in image_records.items()
                             if path in image_info and Path(path).relative_to(root).parts[0].startswith("classification_task")}
    segmentation_paths = {path for path in overlap}
    return {"root": str(root), "task_directory_status": {
                name: {"exists": (root / name).is_dir(), "alternate_exists": (root / f"{name} (1)").is_dir()}
                for name in ("classification_task", "segmentation_task")},
            "path_resolution_counts": dict(resolver.resolution_counts),
            "manifest_rows": len(rows), "image_rows": sum(not _truth(r["is_mask"]) for r in rows),
            "mask_rows": sum(_truth(r["is_mask"]) for r in rows), "split_column": split_col,
            "split_counts": {s: sum(r[split_col].strip().casefold() == s for r in rows) for s in sorted({r[split_col].strip().casefold() for r in rows})},
            "task_split_counts": task_split_counts, "mask_split_counts": mask_split_counts,
            "task_label_counts": task_label_counts,
            "labels_by_task": {k: sorted(v) for k, v in labels_by_task.items()},
            "plane_labels": sorted(planes), "image_modes": sorted(image_modes),
            "mask_pixel_values": sorted(mask_values), "missing_files": missing, "ambiguous_files": ambiguous,
            "corrupt_files": corrupt, "duplicate_references": duplicate_references, "mismatched_masks": mismatched,
            "duplicate_image_content": duplicates, "duplicate_mask_content": duplicate_mask_content,
            "classification_segmentation_image_overlap": len(classification_paths & segmentation_paths),
            "classification_segmentation_content_overlap": len(task_hashes["classification"] & task_hashes["segmentation"]),
            "valid_image_mask_pairs": valid_pairs,
            "manifest_json": json_metadata, "checksums": checksums, "trace_rows": trace_rows,
            "errors": errors}
