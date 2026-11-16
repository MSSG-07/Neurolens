"""Resumable training/evaluation for segmentation-only volumetric baselines."""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from neurolens.evaluation.segmentation import segmentation_metrics
from neurolens.training.checkpoint import load_checkpoint, save_checkpoint


def segmentation_objective(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Cross-entropy plus the repository's soft multiclass Dice objective."""
    target = target.long()
    ce = F.cross_entropy(logits, target)
    probs = logits.softmax(1)
    onehot = F.one_hot(target, logits.shape[1]).movedim(-1, 1).to(probs.dtype)
    axes = (0, 2, 3, 4)
    dice = 1 - ((2 * (probs * onehot).sum(axes) + 1e-5) /
                (probs.sum(axes) + onehot.sum(axes) + 1e-5)).mean()
    total = ce + dice
    return total, {"cross_entropy": ce, "soft_dice_loss": dice, "total": total}


def _foreground_dice_iou(prediction, target):
    """Fast per-volume validation summary; HD95 is reserved for final evaluation."""
    dices, ious = [], []
    for class_id in (1, 2, 3):
        predicted, truth = prediction == class_id, target == class_id
        intersection = np.logical_and(predicted, truth).sum()
        denominator = predicted.sum() + truth.sum()
        union = np.logical_or(predicted, truth).sum()
        dices.append(float(2 * intersection / denominator) if denominator else 1.0)
        ious.append(float(intersection / union) if union else 1.0)
    return float(np.mean(dices)), float(np.mean(ious))


def _epoch(model, loader, device, optimizer=None, scaler=None, accumulation_steps=1):
    training = optimizer is not None
    model.train(training)
    totals, n, case_dice, case_iou = {}, 0, [], []
    if training: optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(loader):
        image = batch["image3d"].to(device, non_blocking=device.type == "cuda").float()
        target = batch["segmentation"].to(device, non_blocking=device.type == "cuda").long()
        if training:
            with torch.autocast(device_type=device.type, enabled=bool(scaler and scaler.is_enabled())):
                output = model(image)
                logits = output["segmentation_logits"]
                loss, parts = segmentation_objective(logits, target)
                scaled_loss = loss / accumulation_steps
            if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite loss at batch {batch_idx}")
            if scaler and scaler.is_enabled(): scaler.scale(scaled_loss).backward()
            else: scaled_loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(loader):
                if scaler and scaler.is_enabled(): scaler.step(optimizer); scaler.update()
                else: optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            with torch.inference_mode():
                output = model(image)
                logits = output["segmentation_logits"]
                loss, parts = segmentation_objective(logits, target)
        size = image.shape[0]
        for name, value in parts.items(): totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
        if not training:
            prediction = logits.argmax(1).detach().cpu().numpy()
            truth = target.detach().cpu().numpy()
            for pred, actual in zip(prediction, truth):
                dice, iou = _foreground_dice_iou(pred, actual)
                case_dice.append(dice); case_iou.append(iou)
        n += size
    if n == 0: raise ValueError("Segmentation DataLoader produced no samples")
    result = {key: value / n for key, value in totals.items()}
    if not training:
        result.update({"foreground_macro_dice": float(np.nanmean(case_dice)),
                       "foreground_macro_iou": float(np.nanmean(case_iou))})
    return result


def _write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    keys = list(history[0]) if history else ["epoch"]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(history)
    temporary.replace(path)


def fit_segmentation_model(model, train_loader, validation_loader, output_dir, *, epochs: int,
                           learning_rate: float, weight_decay: float, accumulation_steps: int = 1,
                           device="cuda", mixed_precision: bool = True, seed: int = 42,
                           experiment_config=None, resume: bool = True) -> dict:
    """Train one segmentation model with validation selection and atomic resume.

    This API never accesses a test loader. Call ``evaluate_segmentation_model``
    only after model selection is complete.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select CPU explicitly or enable a GPU runtime")
    if epochs < 1 or accumulation_steps < 1: raise ValueError("epochs and accumulation_steps must be positive")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    latest, best = output_dir / "latest.pt", output_dir / "best_validation.pt"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")
    config = {"epochs": epochs, "learning_rate": learning_rate, "weight_decay": weight_decay,
              "accumulation_steps": accumulation_steps, "device": device.type,
              "mixed_precision": scaler.is_enabled(), "seed": seed, **(experiment_config or {})}
    history, start_epoch, best_loss, best_epoch = [], 0, math.inf, None
    elapsed_before = 0.0
    if resume and latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state.get("config") != config:
            raise ValueError(f"Refusing incompatible resume for {latest}; saved and requested configs differ")
        latest_epoch = int(state["epoch"])
        if best.is_file():
            best_state = torch.load(best, map_location="cpu", weights_only=False)
            best_epoch_in_file = int(best_state["epoch"])
            if best_state.get("config") != config:
                raise ValueError(f"Best checkpoint configuration mismatch: {best}")
            # A disconnect after writing a newly selected best but before latest
            # is committed leaves best exactly one epoch ahead. Promote it as
            # the canonical recovery state instead of retraining that epoch.
            if best_epoch_in_file > latest_epoch:
                if best_epoch_in_file != latest_epoch + 1:
                    raise RuntimeError(f"Best checkpoint {best} is unexpectedly ahead of latest {latest}")
                temporary = latest.with_suffix(latest.suffix + ".recover.tmp")
                shutil.copyfile(best, temporary); temporary.replace(latest)
                state = best_state; latest_epoch = best_epoch_in_file
        start_epoch = load_checkpoint(latest, model, optimizer, scaler, map_location="cpu")
        extra = state.get("extra_state", {})
        history, best_loss, best_epoch = extra.get("history", []), float(extra.get("best_loss", math.inf)), extra.get("best_epoch")
        if best_epoch is not None:
            if not best.is_file():
                if int(best_epoch) != start_epoch:
                    raise FileNotFoundError(f"Best checkpoint {best} is missing and cannot be recovered from latest {latest}")
                save_checkpoint(best, model, optimizer, scaler, start_epoch, config, extra)
            else:
                best_state = torch.load(best, map_location="cpu", weights_only=False)
                if int(best_state["epoch"]) != int(best_epoch):
                    if int(best_epoch) != start_epoch:
                        raise RuntimeError(f"Best checkpoint epoch disagrees with recovery state: {best}")
                    save_checkpoint(best, model, optimizer, scaler, start_epoch, config, extra)
        elapsed_before = float(extra.get("elapsed_seconds", 0.0))
        if extra.get("python_rng") is not None: random.setstate(extra["python_rng"])
        if extra.get("numpy_rng") is not None: np.random.set_state(extra["numpy_rng"])
        if extra.get("cuda_rng") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(extra["cuda_rng"])
        if start_epoch > epochs: raise ValueError(f"Checkpoint epoch {start_epoch} exceeds requested epoch budget {epochs}")
        if start_epoch >= epochs:
            if not best.is_file(): raise FileNotFoundError(f"Completed checkpoint is missing selected model: {best}")
            _write_history(output_dir / "training.csv", history)
            from neurolens.training.experiment_registry import atomic_json
            metadata_path = output_dir / "run_metadata.json"
            prior_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            atomic_json(metadata_path, {**prior_metadata, "status":"complete", "epochs_completed":start_epoch,
                "best_epoch":best_epoch,"best_validation_loss":best_loss,"elapsed_seconds":elapsed_before,
                "device":torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "peak_gpu_memory_gib":prior_metadata.get("peak_gpu_memory_gib"),"config":config})
            return {"status": "complete", "epochs_completed": start_epoch, "best_epoch": best_epoch,
                    "best_validation_loss": best_loss, "checkpoint": str(best), "history": history}
    else:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    started = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        train_metrics = _epoch(model, train_loader, device, optimizer, scaler, accumulation_steps)
        val_metrics = _epoch(model, validation_loader, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"validation_{k}": v for k, v in val_metrics.items()}}
        if not all(math.isfinite(float(value)) for key, value in row.items() if key != "epoch"):
            raise FloatingPointError(f"Non-finite baseline metrics at epoch {epoch}: {row}")
        history = [item for item in history if int(item["epoch"]) != epoch] + [row]
        improved = val_metrics["total"] < best_loss
        if improved: best_loss, best_epoch = val_metrics["total"], epoch
        elapsed = elapsed_before + time.time() - started
        extra = {"history": history, "best_loss": best_loss, "best_epoch": best_epoch,
                 "elapsed_seconds": elapsed, "python_rng": random.getstate(),
                 "numpy_rng": np.random.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        if improved: save_checkpoint(best, model, optimizer, scaler, epoch, config, extra)
        save_checkpoint(latest, model, optimizer, scaler, epoch, config, extra)
        _write_history(output_dir / "training.csv", history)
        from neurolens.training.experiment_registry import atomic_json
        atomic_json(output_dir / "run_metadata.json", {"status": "running", "epochs_completed": epoch,
                    "best_epoch": best_epoch, "best_validation_loss": best_loss,
                    "elapsed_seconds": elapsed, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                    "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
                    "config": config})
    return {"status": "complete", "epochs_completed": epochs, "best_epoch": best_epoch,
            "best_validation_loss": best_loss, "checkpoint": str(best), "history": history}


def evaluate_segmentation_model(model, loader, device="cuda", output_path=None, artifact_dir=None,
                               qualitative_dir=None, slice_axis: int = 2) -> dict:
    """Evaluate patient-wise segmentation metrics; never selects a checkpoint."""
    device = torch.device(device); model.to(device).eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image3d"].to(device, non_blocking=device.type == "cuda").float()
            prediction = model(image)["segmentation_logits"].argmax(1).cpu().numpy()
            target = batch["segmentation"].cpu().numpy()
            patients = batch.get("patient_id", [str(len(rows) + i) for i in range(len(prediction))])
            for sample_index, (patient, pred, truth) in enumerate(zip(patients, prediction, target)):
                class_metrics = {}
                for class_id, name in ((1, "NCR"), (2, "ED"), (3, "ET")):
                    pm, tm = pred == class_id, truth == class_id
                    inter, den = np.logical_and(pm, tm).sum(), pm.sum() + tm.sum()
                    union = np.logical_or(pm, tm).sum()
                    class_metrics[name] = {"dice": float(2 * inter / den) if den else 1.0,
                                           "iou": float(inter / union) if union else 1.0}
                region_metrics = {}
                for region, pred_region, true_region in (
                    ("whole_tumor", pred > 0, truth > 0),
                    ("tumor_core", np.isin(pred, (1, 3)), np.isin(truth, (1, 3))),
                    ("enhancing_tumor", pred == 3, truth == 3),
                ):
                    inter = np.logical_and(pred_region, true_region).sum()
                    denominator = pred_region.sum() + true_region.sum()
                    union = np.logical_or(pred_region, true_region).sum()
                    tp = inter; fn = true_region.sum() - tp; fp = pred_region.sum() - tp
                    tn = true_region.size - tp - fn - fp
                    region_metrics[region] = {
                        "dice": float(2 * tp / denominator) if denominator else 1.0,
                        "iou": float(tp / union) if union else 1.0,
                        "sensitivity": float(tp / (tp + fn)) if tp + fn else 1.0,
                        "specificity": float(tn / (tn + fp)) if tn + fp else 1.0,
                    }
                rows.append({"patient_id": str(patient), "per_class": class_metrics,
                             "tumor_regions": region_metrics,
                             "hd95_macro_voxel_units": segmentation_metrics(pred, truth, spacing=(1., 1., 1.))["hd95"],
                             "foreground_macro_dice": float(np.mean([m["dice"] for m in class_metrics.values()])),
                             "foreground_macro_iou": float(np.mean([m["iou"] for m in class_metrics.values()]))})
                if artifact_dir is not None:
                    folder = Path(artifact_dir); folder.mkdir(parents=True, exist_ok=True)
                    artifact = folder / f"{patient}.npz"; temporary = artifact.with_suffix(".npz.tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, patient_id=np.asarray(str(patient)),
                                            segmentation_prediction=pred.astype(np.uint8),
                                            segmentation_target=truth.astype(np.uint8),
                                            axis_order=np.asarray("(D,H,W)=(Z,Y,X), cropped patch space"))
                    temporary.replace(artifact)
                if qualitative_dir is not None:
                    if slice_axis not in (0, 1, 2): raise ValueError("slice_axis must be 0, 1, or 2")
                    slice_index = int(batch["slice_index"][sample_index])
                    image_slice = batch["image2d"][sample_index, 0].cpu().numpy()
                    pred_slice = np.take(pred, slice_index, axis=slice_axis)
                    truth_slice = np.take(truth, slice_index, axis=slice_axis)
                    import matplotlib.pyplot as plt
                    from pathlib import Path
                    folder = Path(qualitative_dir); folder.mkdir(parents=True, exist_ok=True)
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    axes[0].imshow(image_slice, cmap="gray"); axes[0].set_title("MRI")
                    axes[1].imshow(truth_slice, cmap="tab10", vmin=0, vmax=3); axes[1].set_title("Ground truth")
                    axes[2].imshow(pred_slice, cmap="tab10", vmin=0, vmax=3); axes[2].set_title("Prediction")
                    axes[3].imshow(image_slice, cmap="gray"); axes[3].imshow(pred_slice, cmap="tab10", alpha=.45, vmin=0, vmax=3); axes[3].set_title("Overlay")
                    for axis in axes: axis.axis("off")
                    fig.suptitle(str(patient)); fig.tight_layout()
                    output = folder / f"{patient}_segmentation.png"
                    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
                    fig.savefig(temporary, dpi=140); plt.close(fig); temporary.replace(output)
    if not rows: raise ValueError("Evaluation DataLoader produced no patients")
    result = {"patient_count": len(rows), "cases": rows,
              "foreground_macro_dice": float(np.mean([r["foreground_macro_dice"] for r in rows])),
              "foreground_macro_iou": float(np.mean([r["foreground_macro_iou"] for r in rows])),
              "aggregation": "foreground classes macro-averaged within each patient, then averaged across patients",
              "hd95_assumption": "unit voxel spacing because the preprocessing resamples/crops into patch coordinates without physical-affine spacing; values are voxels, not millimetres",
              "tumor_region_definition": {"whole_tumor":"labels 1|2|3", "tumor_core":"labels 1|3", "enhancing_tumor":"label 3 (source label 4)"}}
    if output_path:
        from neurolens.training.experiment_registry import atomic_json
        atomic_json(output_path, result)
    return result
