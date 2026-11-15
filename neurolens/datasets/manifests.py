"""Read split manifests in place; these helpers never rewrite manifest files."""
import csv
import json
from pathlib import Path
from typing import Any


ID_COLUMNS = ("patient_id", "case_id", "subject_id", "BraTSID", "participant_id", "id")


def inspect_manifest(path: str) -> dict[str, Any]:
    """Return format/header/sample metadata for a manifest without changing it."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Manifest not found: {file}")
    suffix = file.suffix.lower()
    if suffix == ".csv":
        with file.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        return {"path": str(file), "format": "csv", "columns": reader.fieldnames or [],
                "row_count": len(rows), "sample": rows[:3]}
    if suffix == ".json":
        with file.open(encoding="utf-8") as stream:
            value = json.load(stream)
        return {"path": str(file), "format": "json", "top_level_type": type(value).__name__,
                "keys": list(value)[:30] if isinstance(value, dict) else None,
                "row_count": len(value) if isinstance(value, (list, dict)) else None,
                "sample": value[:3] if isinstance(value, list) else value}
    if suffix in {".txt", ".list"}:
        lines = [line.strip() for line in file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        return {"path": str(file), "format": "lines", "row_count": len(lines), "sample": lines[:3]}
    raise ValueError(f"Unsupported manifest format {suffix!r}; expected CSV, JSON, or TXT")


def _extract_json_ids(value: Any, id_column: str) -> list[str]:
    if isinstance(value, list):
        if all(isinstance(item, (str, int)) for item in value):
            return [str(item) for item in value]
        if all(isinstance(item, dict) for item in value):
            for key in (id_column, *ID_COLUMNS):
                if all(key in row for row in value):
                    return [str(row[key]) for row in value]
    if isinstance(value, dict):
        for key in ("patient_ids", "patients", "case_ids", "cases", "data", "samples"):
            if key in value:
                return _extract_json_ids(value[key], id_column)
    raise ValueError("Cannot identify patient IDs in JSON manifest; inspect it and set a supported schema")


def read_patient_ids(path: str, id_column: str = "patient_id") -> list[str]:
    """Load manifest IDs in recorded order, without altering or regenerating the split."""
    file = Path(path)
    info = inspect_manifest(str(file))
    if info["format"] == "lines":
        values = info["sample"]
        # Read all rows after inspection, preserving their order.
        values = [line.strip() for line in file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif info["format"] == "csv":
        with file.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            column = id_column if id_column in fields else next((key for key in ID_COLUMNS if key in fields), None)
            if column is None:
                raise ValueError(f"No patient ID column found in {file}; columns={fields}; configure manifest_id_column")
            values = [row[column] for row in reader]
    else:
        with file.open(encoding="utf-8") as stream:
            values = _extract_json_ids(json.load(stream), id_column)
    ids = [str(value).strip() for value in values if str(value).strip()]
    if not ids:
        raise ValueError(f"Manifest contains no patient IDs: {file}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Manifest contains duplicate patient IDs: {file}")
    return ids
