"""Audit and load BRISC manifest samples; never trains or changes model code."""
import argparse
import json

from neurolens.datasets.brisc import BRISCDataset, validate_brisc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="BRISC dataset root")
    parser.add_argument("--manifest", default="", help="Defaults to <root>/manifest.csv")
    args = parser.parse_args()
    manifest = args.manifest or None
    audit = validate_brisc(args.root, manifest)
    print(json.dumps(audit, indent=2))
    if audit["missing_files"] or audit["mismatched_masks"] or audit["errors"]:
        raise SystemExit("BRISC manifest audit found unresolved file, pair, or metadata errors")
    splits = sorted(audit["split_counts"])
    total = {"classification_samples": 0, "segmentation_samples": 0, "checked_samples": 0}
    for split in splits:
        split = "validation" if split in {"valid", "validation"} else split
        for task in ("classification", "segmentation"):
            dataset = BRISCDataset(args.root, manifest, task=task, split=split)
            for index in range(len(dataset)):
                sample = dataset[index]
                if sample["image"].ndim != 3:
                    raise AssertionError(f"Unexpected image shape {tuple(sample['image'].shape)}")
                if task == "segmentation" and sample["mask"].shape != sample["image"].shape[-2:]:
                    raise AssertionError("Image-mask dimensions differ during load")
            key = f"{task}_samples"
            total[key] += len(dataset)
            total["checked_samples"] += len(dataset)
    print(json.dumps({"status": "BRISC manifest integration passed", **total}, indent=2))


if __name__ == "__main__":
    main()
