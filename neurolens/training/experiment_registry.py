"""Small atomic experiment registry and prediction-cache helpers."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _json_safe(value):
    if isinstance(value, Mapping): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return _json_safe(value.tolist())
    if isinstance(value, np.generic): return _json_safe(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def atomic_json(path, value: Mapping[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def update_experiment_registry(path, experiment_id: str, record: Mapping[str, Any]) -> dict:
    """Atomically upsert one experiment record without deleting other entries."""
    path = Path(path)
    registry = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"experiments": {}}
    if not isinstance(registry.get("experiments"), dict):
        raise ValueError(f"Malformed experiment registry at {path}: 'experiments' must be an object")
    existing = registry["experiments"].get(experiment_id, {})
    registry["experiments"][experiment_id] = {**existing, **dict(record), "experiment_id": experiment_id}
    atomic_json(path, registry)
    return registry["experiments"][experiment_id]


def save_prediction_cache(path, patient_ids, probabilities, metadata: Mapping[str, Any], labels=None) -> None:
    """Write aligned per-patient predictions atomically; labels are opt-in.

    OOF training caches may include their train labels. Validation/test caches
    should omit labels until final evaluation so the model-selection code cannot
    accidentally consume held-out targets.
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(list(patient_ids), dtype=str)
    probs = np.asarray(probabilities)
    if probs.shape[0] != len(ids): raise ValueError("Prediction count does not match patient_ids")
    if len(set(ids.tolist())) != len(ids): raise ValueError("Duplicate patient IDs in prediction cache")
    if not np.isfinite(probs).all(): raise ValueError("Prediction cache contains non-finite values")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        payload = {"patient_ids": ids, "probabilities": probs,
                   "metadata": np.asarray(json.dumps(dict(metadata), sort_keys=True))}
        if labels is not None:
            labels = np.asarray(labels)
            if labels.shape[0] != len(ids): raise ValueError("Label count does not match patient_ids")
            payload["labels"] = labels
        np.savez_compressed(stream, **payload)
    os.replace(tmp, path)


def load_prediction_cache(path, expected_patient_ids=None, expected_metadata=None) -> dict:
    """Load cache and enforce patient order and model/preprocessing provenance."""
    with np.load(Path(path), allow_pickle=False) as cache:
        ids = cache["patient_ids"].astype(str).tolist()
        probabilities = cache["probabilities"]
        metadata = json.loads(str(cache["metadata"].item()))
        labels = cache["labels"] if "labels" in cache.files else None
    if expected_patient_ids is not None and ids != list(expected_patient_ids):
        raise ValueError("Prediction cache patient IDs/order do not match the requested split")
    if expected_metadata is not None:
        mismatch = {key: (metadata.get(key), value) for key, value in expected_metadata.items()
                    if metadata.get(key) != value}
        if mismatch: raise ValueError(f"Prediction cache provenance mismatch: {mismatch}")
    return {"patient_ids": ids, "probabilities": probabilities, "metadata": metadata, "labels": labels}


def save_multimodel_probability_cache(path, patient_ids, probabilities: Mapping[str, np.ndarray],
                                      labels: np.ndarray, metadata: Mapping[str, Any],
                                      classification_probabilities: Mapping[str, np.ndarray] | None = None,
                                      paired_cosines: Mapping[str, np.ndarray] | None = None) -> None:
    """Atomically save aligned 3D class probabilities for several models.

    Model arrays must all be ``(N,C,D,H,W)`` and share identical spatial grids.
    Segmentation labels are stored for validation fusion and, in a separately
    generated post-selection cache, final test evaluation.
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(list(patient_ids), dtype=str)
    if not len(ids) or len(set(ids.tolist())) != len(ids):
        raise ValueError("patient_ids must be non-empty and unique")
    labels = np.asarray(labels)
    if labels.ndim != 4 or labels.shape[0] != len(ids):
        raise ValueError("labels must have shape (N,D,H,W) matching patient_ids")
    payload = {"patient_ids": ids, "labels": labels.astype(np.uint8, copy=False),
               "metadata": np.asarray(json.dumps(_json_safe(dict(metadata)), sort_keys=True, allow_nan=False))}
    reference_shape = None
    for index, (name, values) in enumerate(probabilities.items()):
        array = np.asarray(values)
        if array.ndim != 5 or array.shape[0] != len(ids):
            raise ValueError(f"Model {name} probabilities must have shape (N,C,D,H,W)")
        if reference_shape is None: reference_shape = array.shape
        if array.shape != reference_shape or array.shape[2:] != labels.shape[1:]:
            raise ValueError(f"Model {name} probabilities are not spatially aligned with labels/other models")
        if not np.isfinite(array).all(): raise ValueError(f"Model {name} probabilities contain non-finite values")
        if (array < 0).any() or (array > 1).any(): raise ValueError(f"Model {name} values are outside probability range [0,1]")
        if not np.allclose(array.sum(axis=1), 1.0, atol=2e-3): raise ValueError(f"Model {name} class probabilities do not sum to one")
        payload[f"prob_{index}"] = array.astype(np.float16, copy=False)
    class_names = list((classification_probabilities or {}).keys())
    for index, name in enumerate(class_names):
        array = np.asarray(classification_probabilities[name], dtype=np.float32)
        if array.shape != (len(ids), 2) or not np.isfinite(array).all():
            raise ValueError(f"Classification probabilities for {name} must have shape (N,2) and be finite")
        if (array < 0).any() or (array > 1).any() or not np.allclose(array.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError(f"Classification probabilities for {name} are invalid")
        payload[f"class_prob_{index}"] = array.astype(np.float32, copy=False)
    for name, values in (paired_cosines or {}).items():
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (len(ids),) or not np.isfinite(array).all():
            raise ValueError(f"Paired cosine values for {name} must have shape (N,) and be finite")
        payload[f"paired_cosine_{class_names.index(name)}"] = array
    payload["metadata"] = np.asarray(json.dumps({**_json_safe(dict(metadata)),
        "base_model_names":list(probabilities), "classification_model_names":class_names,
        "paired_cosine_model_names":list((paired_cosines or {}).keys())}, sort_keys=True, allow_nan=False))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream: np.savez_compressed(stream, **payload)
    os.replace(tmp, path)


def load_multimodel_probability_cache(path, expected_patient_ids=None, expected_metadata=None) -> dict:
    """Load a multimodel volume cache and check patient order/provenance."""
    with np.load(Path(path), allow_pickle=False) as archive:
        ids = archive["patient_ids"].astype(str).tolist()
        labels = archive["labels"].copy()
        metadata = json.loads(str(archive["metadata"].item()))
        names = metadata.get("base_model_names", [])
        probabilities = {name: archive[f"prob_{index}"].astype(np.float32)
                         for index, name in enumerate(names)}
        class_names = metadata.get("classification_model_names", [])
        class_probabilities = {name:archive[f"class_prob_{index}"].astype(np.float32)
                               for index,name in enumerate(class_names)}
        cosine_names = metadata.get("paired_cosine_model_names", [])
        paired_cosines = {name:archive[f"paired_cosine_{class_names.index(name)}"].astype(np.float32)
                          for name in cosine_names}
    if expected_patient_ids is not None and ids != list(expected_patient_ids):
        raise ValueError("Multimodel cache patient IDs/order do not match the requested split")
    if expected_metadata is not None:
        mismatch = {key: (metadata.get(key), value) for key, value in expected_metadata.items()
                    if metadata.get(key) != value}
        if mismatch: raise ValueError(f"Multimodel cache provenance mismatch: {mismatch}")
    return {"patient_ids":ids,"labels":labels,"metadata":metadata,"probabilities":probabilities,
            "classification_probabilities":class_probabilities,"paired_cosines":paired_cosines}
