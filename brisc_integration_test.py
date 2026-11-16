"""Audit and load BRISC manifest samples; never trains or changes model code."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="BRISC dataset root")
    parser.add_argument("--manifest", default="", help="Defaults to <root>/manifest.csv")
    parser.add_argument("--trace-filename", default="", help="Include matching manifest fields and resolver call in audit JSON")
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from neurolens.datasets.brisc import BRISCDataset, validate_brisc

    manifest = args.manifest or None
    audit = validate_brisc(args.root, manifest, trace_filename=args.trace_filename or None)
    print(json.dumps(audit, indent=2))
    if (audit["missing_files"] or audit["ambiguous_files"] or audit["corrupt_files"]
            or audit["mismatched_masks"] or audit["errors"]):
        raise SystemExit("BRISC manifest audit found unresolved file, pair, or metadata errors")
    split_alias = {"valid": "validation", "validation": "validation"}
    splits = sorted({split_alias.get(value, value) for value in audit["split_counts"]})
    total = {"classification_samples": 0, "segmentation_samples": 0, "checked_samples": 0, "batches": {}}
    for split in splits:
        for task in ("classification", "segmentation"):
            dataset = BRISCDataset(args.root, manifest, task=task, split=split)
            for index in range(len(dataset)):
                sample = dataset[index]
                if sample["image"].ndim != 3:
                    raise AssertionError(f"Unexpected image shape {tuple(sample['image'].shape)}")
                if task == "segmentation" and sample["mask"].shape != sample["image"].shape[-2:]:
                    raise AssertionError("Image-mask dimensions differ during load")
                if not torch.isfinite(sample["image"]).all():
                    raise AssertionError(f"Non-finite {task} image in {split}")
            key = f"{task}_samples"
            total[key] += len(dataset)
            total["checked_samples"] += len(dataset)
            batch = next(iter(DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False, num_workers=0)))
            if batch["image"].ndim != 4 or batch["class_label"].dtype != torch.long:
                raise AssertionError(f"Invalid {task} DataLoader batch: {batch['image'].shape}, {batch['class_label'].dtype}")
            if task == "segmentation" and batch["mask"].ndim != 3:
                raise AssertionError(f"Invalid segmentation mask batch shape: {batch['mask'].shape}")
            total["batches"][f"{task}_{split}"] = {
                "image": tuple(batch["image"].shape), "image_dtype": str(batch["image"].dtype),
                "class_label": tuple(batch["class_label"].shape),
                "mask": tuple(batch["mask"].shape) if task == "segmentation" else None,
            }
    print(json.dumps({"status": "BRISC manifest integration passed", **total}, indent=2))


if __name__ == "__main__":
    main()
