"""Run one real paired BraTS batch through forward/loss only; never trains."""
import argparse
import torch
from torch.utils.data import DataLoader
from neurolens.config import load_config
from neurolens.datasets.brats import BraTSPairedDataset, cases_from_manifest, discover_patients
from neurolens.datasets.manifests import read_patient_ids
from neurolens.losses.multitask_loss import MultiTaskLoss
from neurolens.models.multitask import NeuroLens
from neurolens.training.train import validate_saved_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    cfg = load_config(args.config)
    validate_saved_splits(cfg)
    if not cfg.brats_root:
        raise ValueError("Set brats_root or NEUROLENS_BRATS_ROOT before the integration check")
    if not cfg.manifest_train:
        raise ValueError("Set manifest_train or NEUROLENS_MANIFEST_TRAIN before the integration check")
    ids = read_patient_ids(cfg.manifest_train, cfg.manifest_id_column)
    cases = discover_patients(cfg.brats_root, cfg.modalities)
    selected = cases_from_manifest(cases, ids, "train")
    dataset = BraTSPairedDataset(cfg.brats_root, selected, cfg.modalities, cfg.patch_size, cfg.slice_axis)
    if len(dataset) < args.batch_size:
        raise ValueError(f"Need at least {args.batch_size} train-manifest patients for this batch test; found {len(dataset)}")
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    model = NeuroLens(len(cfg.modalities), cfg.base_channels, cfg.projection_dim, cfg.num_classes,
                      cfg.classification_classes).to(device).eval()
    weights = {"classification": cfg.lambda_classification, "segmentation": cfg.lambda_segmentation,
               "contrastive": cfg.lambda_contrastive, "consistency": cfg.lambda_consistency}
    criterion = MultiTaskLoss(weights, cfg.temperature)
    with torch.inference_mode():
        output = model(batch["image2d"].to(device).float(), batch["image3d"].to(device).float())
        loss, terms = criterion(output, {"class_label": batch["class_label"].to(device),
                                        "segmentation": batch["segmentation"].to(device)})
    assert output["embedding2d"].shape == (args.batch_size, cfg.projection_dim)
    assert output["embedding3d"].shape == (args.batch_size, cfg.projection_dim)
    assert output["segmentation_logits"].shape[0] == args.batch_size
    assert torch.isfinite(loss).item() and all(torch.isfinite(value).item() for value in terms.values())
    print({"status": "one real BraTS batch passed", "device": str(device), "patients": batch["patient_id"],
           "image2d": tuple(batch["image2d"].shape), "image3d": tuple(batch["image3d"].shape),
           "segmentation": tuple(batch["segmentation"].shape), "loss_finite": True})


if __name__ == "__main__":
    main()
