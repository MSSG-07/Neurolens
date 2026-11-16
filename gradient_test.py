"""Check real BraTS gradient flow and exactly one isolated AdamW update.

This is a diagnostic script, not a training entry point. It loads one batch from
the saved training manifest, performs one forward/backward/update, and writes no
checkpoints or other artifacts.
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from neurolens.config import load_config
from neurolens.losses.multitask_loss import MultiTaskLoss
from neurolens.models.multitask import NeuroLens
from neurolens.training.train import make_datasets, validate_saved_splits
from neurolens.utils.reproducibility import seed_everything


def _gradient_summary(model, weights):
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    with_grad = [(name, parameter) for name, parameter in parameters if parameter.grad is not None]
    missing = [name for name, parameter in parameters if parameter.grad is None]
    nonfinite = [name for name, parameter in with_grad if not torch.isfinite(parameter.grad).all().item()]

    def branch(prefixes, enabled=True):
        members = [(name, parameter) for name, parameter in parameters
                   if any(name.startswith(prefix) for prefix in prefixes)]
        grad_members = [(name, parameter) for name, parameter in members if parameter.grad is not None]
        finite_members = [(name, parameter) for name, parameter in grad_members
                          if torch.isfinite(parameter.grad).all().item()]
        norm_sq = sum(float(parameter.grad.detach().float().pow(2).sum().item())
                      for _, parameter in finite_members)
        status = "disabled_by_config" if not enabled else (
            "passed" if members and len(grad_members) == len(members) and len(finite_members) == len(members)
            else "failed"
        )
        return {
            "status": status,
            "expected_active": enabled,
            "parameter_tensors": len(members),
            "parameter_tensors_with_grad": len(grad_members),
            "parameter_tensors_with_nonfinite_grad": len(grad_members) - len(finite_members),
            "gradient_l2_norm": norm_sq ** 0.5,
            "missing_parameters": [name for name, parameter in members if parameter.grad is None],
            "nonfinite_parameters": [name for name, parameter in grad_members
                                     if not torch.isfinite(parameter.grad).all().item()],
        }

    alignment_enabled = any((
        weights["contrastive"] != 0,
        weights["consistency"] != 0,
    ))
    branches = {
        "encoder2d_backbone": branch(("encoder2d.features.",), weights["classification"] != 0 or alignment_enabled),
        "classification_head": branch(("encoder2d.classifier.",), weights["classification"] != 0),
        "encoder2d_projection": branch(("encoder2d.projection.",), alignment_enabled),
        "encoder3d_encoder": branch(("encoder3d.enc1.", "encoder3d.enc2.", "encoder3d.enc3.", "encoder3d.enc4."),
                                    weights["segmentation"] != 0 or alignment_enabled),
        "encoder3d_projection": branch(("encoder3d.projection.",), alignment_enabled),
        "cross_dimensional_projection_heads": branch(("project2d.", "project3d."), alignment_enabled),
        "segmentation_decoder": branch(("encoder3d.up3.", "encoder3d.dec3.", "encoder3d.up2.",
                                         "encoder3d.dec2.", "encoder3d.up1.", "encoder3d.dec1.",
                                         "encoder3d.segmentation_head."),
                                        weights["segmentation"] != 0),
    }
    return {
        "trainable_parameter_tensors": len(parameters),
        "trainable_parameters": sum(parameter.numel() for _, parameter in parameters),
        "parameter_tensors_with_grad": len(with_grad),
        "parameter_tensors_missing_grad": len(missing),
        "parameters_missing_grad": missing,
        "parameter_tensors_with_nonfinite_grad": len(nonfinite),
        "parameters_with_nonfinite_grad": nonfinite,
        "gradient_l2_norm": sum(float(parameter.grad.detach().float().pow(2).sum().item())
                                 for _, parameter in with_grad if torch.isfinite(parameter.grad).all().item()) ** 0.5,
        "branches": branches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Defaults to configured batch_size; must be at least two for aligned loss.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_counts = validate_saved_splits(cfg)
    batch_size = args.batch_size or cfg.batch_size
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if batch_size < 2 and cfg.alignment_condition in ("B", "C") and cfg.lambda_contrastive != 0:
        raise ValueError("Aligned gradient test requires batch_size >= 2 for InfoNCE negatives")
    seed_everything(cfg.seed)

    train_dataset, _ = make_datasets(cfg)
    if len(train_dataset) < batch_size:
        raise ValueError(f"Need {batch_size} saved training cases, found {len(train_dataset)}")
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    batch = next(iter(loader))

    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = NeuroLens(len(cfg.modalities), cfg.base_channels, cfg.projection_dim, cfg.num_classes,
                      cfg.classification_classes).to(device)
    model.train()

    weights = {
        "classification": cfg.lambda_classification,
        "segmentation": cfg.lambda_segmentation,
        "contrastive": cfg.lambda_contrastive if cfg.alignment_condition in ("B", "C") else 0.0,
        "consistency": cfg.lambda_consistency if cfg.alignment_condition == "C" else 0.0,
    }
    criterion = MultiTaskLoss(weights, cfg.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    image2d = batch["image2d"].to(device=device, dtype=torch.float32)
    image3d = batch["image3d"].to(device=device, dtype=torch.float32)
    targets = {
        "class_label": batch["class_label"].to(device=device, dtype=torch.long),
        "segmentation": batch["segmentation"].to(device=device, dtype=torch.long),
    }
    output = model(image2d, image3d)
    loss, terms = criterion(output, targets)
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"Non-finite multitask loss: {float(loss.detach())}")
    loss.backward()

    gradients = _gradient_summary(model, weights)
    params_with_finite_nonzero_grad = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        and torch.count_nonzero(parameter.grad).item() > 0
    ]
    if not params_with_finite_nonzero_grad:
        raise RuntimeError("No trainable parameter received a finite, nonzero gradient; optimizer update not attempted")
    watched_parameter = params_with_finite_nonzero_grad[0]
    watched_before = watched_parameter.detach().clone()

    optimizer.step()
    optimizer_updates = 1
    updated_parameters_finite = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
    watched_parameter_changed = not torch.equal(watched_before, watched_parameter.detach())
    optimizer_status = "passed" if updated_parameters_finite and watched_parameter_changed else "failed"

    branch_failures = [name for name, info in gradients["branches"].items() if info["status"] == "failed"]
    gradient_status = "passed" if not gradients["parameters_with_nonfinite_grad"] and not branch_failures else "failed"
    report = {
        "status": "passed" if gradient_status == "passed" and optimizer_status == "passed" else "failed",
        "dataset": {
            "source": "saved BraTS training manifest",
            "split_counts": split_counts,
            "training_cases": len(train_dataset),
            "patient_ids": batch["patient_id"],
            "image2d_shape": tuple(batch["image2d"].shape),
            "image3d_shape": tuple(batch["image3d"].shape),
            "segmentation_shape": tuple(batch["segmentation"].shape),
            "image2d_dtype": str(batch["image2d"].dtype),
            "image3d_dtype": str(batch["image3d"].dtype),
            "segmentation_dtype": str(batch["segmentation"].dtype),
        },
        "device": str(device),
        "model_mode": "train",
        "mixed_precision": False,
        "loss_weights": weights,
        "loss_components": {name: float(value.detach().item()) for name, value in terms.items()},
        "gradients": gradients,
        "gradient_status": gradient_status,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "updates_performed": optimizer_updates,
            "updated_parameters_finite": updated_parameters_finite,
            "watched_parameter_changed": watched_parameter_changed,
            "status": optimizer_status,
            "checkpoint_written": False,
        },
        "branch_failures": branch_failures,
    }
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
