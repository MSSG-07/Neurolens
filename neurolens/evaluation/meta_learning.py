"""Leakage-aware prediction stacking and probability-fusion utilities.

The preferred classifier inputs are training-derived out-of-fold (OOF)
probabilities; a clearly marked train-only in-sample fallback is also supported.
Validation predictions select among candidates. Test labels are not accepted
by the fit/selection API and are reserved for a separate final evaluation call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np


def _check_probabilities(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError(f"{name} must have shape (patients, base_models, 2), got {values.shape}")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"{name} contains invalid probabilities")
    if not np.allclose(values.sum(axis=2), 1.0, atol=1e-4):
        raise ValueError(f"Each class probability pair in {name} must sum to one")
    return values


@dataclass
class ClassificationStack:
    models: Dict[str, Any]
    selected_name: str
    selected_model: Any
    validation_scores: Dict[str, float]
    base_model_names: tuple
    training_prediction_mode: str

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        x = _check_probabilities(probabilities, "inference probabilities")
        if x.shape[1] != len(self.base_model_names): raise ValueError("Base-model order/count differs from fitted stack")
        features = x.reshape(x.shape[0], -1)
        if self.selected_name == "mean_probability": return x.mean(axis=1)
        if self.selected_name == "best_individual": return x[:, int(self.selected_model), :]
        if self.selected_name == "weighted_probability":
            return np.einsum("m,nmc->nc", self.selected_model, x)
        return self.selected_model.predict_proba(features)


def fit_classification_stack(train_probabilities: np.ndarray, train_labels: Iterable[int],
                             validation_probabilities: np.ndarray, validation_labels: Iterable[int],
                             candidates=("logistic_regression", "random_forest", "mean_probability", "weighted_probability"),
                             random_state: int = 42, base_model_names=None,
                             training_prediction_mode: str = "oof") -> ClassificationStack:
    """Fit meta candidates on train-only predictions; select by validation macro-F1.

    Inputs are explicit so callers cannot accidentally pass final-test labels.
    Prefer ``training_prediction_mode='oof'`` with patient-level folds. The
    documented compute-limited alternative ``'in_sample_train'`` is accepted
    but can overfit because base learners have seen those patients. In either
    mode, test labels are not accepted by this fitting/selection API.
    """
    if training_prediction_mode not in {"oof", "in_sample_train"}:
        raise ValueError("training_prediction_mode must be 'oof' or 'in_sample_train'")
    train_x = _check_probabilities(train_probabilities, "train_probabilities")
    val_x = _check_probabilities(validation_probabilities, "validation_probabilities")
    train_y, val_y = np.asarray(list(train_labels), dtype=int), np.asarray(list(validation_labels), dtype=int)
    if len(train_y) != len(train_x) or len(val_y) != len(val_x): raise ValueError("Probability/label row counts differ")
    if set(np.unique(train_y)) - {0, 1} or set(np.unique(val_y)) - {0, 1}: raise ValueError("Expected binary tumor-presence labels {0,1}")
    if train_x.shape[1] != val_x.shape[1]: raise ValueError("Train/validation base-model counts differ")
    model_names = tuple(base_model_names) if base_model_names is not None else tuple(f"model_{i}" for i in range(train_x.shape[1]))
    if len(model_names) != train_x.shape[1] or len(set(model_names)) != len(model_names):
        raise ValueError("base_model_names must contain one unique name per model axis")
    from sklearn.metrics import f1_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    X, Xv = train_x.reshape(len(train_x), -1), val_x.reshape(len(val_x), -1)
    fitted, scores = {}, {}
    for name in candidates:
        if name == "mean_probability":
            fitted[name] = None; prediction = val_x.mean(axis=1).argmax(axis=1)
        elif name == "best_individual":
            individual_scores = [float(f1_score(val_y, val_x[:, index, :].argmax(axis=1), average="macro", zero_division=0))
                                 for index in range(val_x.shape[1])]
            selected_index = int(np.argmax(individual_scores))
            fitted[name] = selected_index
            prediction = val_x[:, selected_index, :].argmax(axis=1)
        elif name == "weighted_probability":
            # Small deterministic validation grid; no test labels are accepted.
            m = val_x.shape[1]
            if m > 6: raise ValueError("Weighted probability search supports at most six base models")
            import itertools
            weight_grid = sorted({tuple(float(v) for v in combo) for combo in itertools.product((0., .5, 1.), repeat=m)
                                  if sum(combo) > 0})
            best_weights, best_score, prediction = None, -1.0, None
            for combo in weight_grid:
                w = np.asarray(combo, dtype=np.float64); w /= w.sum()
                pred = np.einsum("m,nmc->nc", w, val_x).argmax(axis=1)
                score = float(f1_score(val_y, pred, average="macro", zero_division=0))
                if score > best_score: best_weights, best_score, prediction = w, score, pred
            fitted[name] = best_weights
        else:
            if name == "logistic_regression": estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
            elif name == "random_forest": estimator = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1)
            elif name == "xgboost":
                try: from xgboost import XGBClassifier
                except ImportError: continue
                estimator = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.03, subsample=.8,
                                          colsample_bytree=.8, eval_metric="logloss", random_state=random_state)
            elif name == "lightgbm":
                try: from lightgbm import LGBMClassifier
                except ImportError: continue
                estimator = LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=.03, random_state=random_state, verbosity=-1)
            else: raise ValueError(f"Unknown stack candidate: {name}")
            estimator.fit(X, train_y)
            fitted[name] = estimator
            prediction = estimator.predict(Xv)
        scores[name] = float(f1_score(val_y, prediction, average="macro", zero_division=0))
    if not scores: raise RuntimeError("No requested meta-learner could be fitted")
    selected = max(scores, key=scores.get)
    return ClassificationStack(fitted, selected, fitted[selected], scores, model_names, training_prediction_mode)


def select_segmentation_weights(validation_probabilities: Mapping[str, np.ndarray],
                                validation_labels: np.ndarray,
                                candidates: Optional[Mapping[str, float]] = None) -> dict:
    """Select coarse fusion weights using validation foreground macro Dice.

    Each probability volume is ``(N,C,D,H,W)`` on the same grid. Candidate
    weight maps are supplied explicitly; the default compares equal weights and
    each single-model predictor. No test labels enter selection.
    """
    if not validation_probabilities: raise ValueError("No validation model probabilities provided")
    names = list(validation_probabilities)
    arrays = [np.asarray(validation_probabilities[n], dtype=np.float32) for n in names]
    shape = arrays[0].shape
    if len(shape) != 5 or any(a.shape != shape for a in arrays): raise ValueError("All probabilities must share shape (N,C,D,H,W)")
    if not all(np.isfinite(a).all() for a in arrays): raise ValueError("Validation probabilities contain non-finite values")
    truth = np.asarray(validation_labels)
    if truth.shape != (shape[0], *shape[2:]): raise ValueError("Validation masks do not match prediction grids")
    weight_sets = dict(candidates or {"equal": {name: 1.0 for name in names}, **{f"only_{name}": {name: 1.0} for name in names}})
    scores = {}
    for key, raw_weights in weight_sets.items():
        weights = np.asarray([float(raw_weights.get(name, 0.0)) for name in names])
        if (weights < 0).any() or weights.sum() <= 0: raise ValueError(f"Invalid fusion weights for {key}")
        fused = sum(w * a for w, a in zip(weights / weights.sum(), arrays)).argmax(axis=1)
        per_case = []
        for pred, target in zip(fused, truth):
            class_scores = []
            for cls in (1, 2, 3):
                p, t = pred == cls, target == cls
                den = p.sum() + t.sum()
                class_scores.append(2 * np.logical_and(p, t).sum() / den if den else 1.0)
            per_case.append(float(np.mean(class_scores)))
        scores[key] = {"foreground_macro_dice": float(np.mean(per_case)),
                       "weights": {name: float(w / weights.sum()) for name, w in zip(names, weights)}}
    selected = max(scores, key=lambda key: scores[key]["foreground_macro_dice"])
    return {"selected": selected, "candidates": scores,
            "selection_split": "validation", "patient_macro": True}


def evaluate_classification_stack(stack: ClassificationStack, probabilities: np.ndarray) -> np.ndarray:
    """Create predictions after a selection is frozen; labels are intentionally absent."""
    return stack.predict_proba(probabilities)
