"""Manifest-driven BraTS dataset construction; no splits are regenerated."""
from neurolens.datasets.brats import discover_patients, cases_from_manifest, BraTSPairedDataset
from neurolens.datasets.manifests import read_patient_ids


def _dataset_for_manifest(config, manifest_path, split_name, cases=None):
    if not manifest_path:
        raise ValueError(f"Configure the saved {split_name} manifest path in config.yaml or environment")
    ids = read_patient_ids(manifest_path, config.manifest_id_column)
    available_cases = cases if cases is not None else discover_patients(config.brats_root or config.data_root, config.modalities)
    selected = cases_from_manifest(available_cases, ids, split_name)
    return BraTSPairedDataset(config.brats_root or config.data_root, selected, modalities=config.modalities,
                              patch_size=config.patch_size, slice_axis=config.slice_axis,
                              random_slice=(split_name == "train"))


def validate_saved_splits(config):
    """Check saved split files for overlap without modifying or regenerating them."""
    paths = {"train": config.manifest_train, "validation": config.manifest_val, "test": config.manifest_test}
    ids = {}
    for split, path in paths.items():
        if not path:
            raise ValueError(f"Configure the saved {split} manifest path in config.yaml or environment")
        ids[split] = set(read_patient_ids(path, config.manifest_id_column))
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = ids[left] & ids[right]
        if overlap:
            raise ValueError(f"Saved {left}/{right} manifests overlap for {len(overlap)} patients; examples: {sorted(overlap)[:5]}")
    return {split: len(split_ids) for split, split_ids in ids.items()}


def make_datasets(config):
    """Return the saved train and validation manifests exactly as assigned."""
    validate_saved_splits(config)
    cases = discover_patients(config.brats_root or config.data_root, config.modalities)
    return (_dataset_for_manifest(config, config.manifest_train, "train", cases),
            _dataset_for_manifest(config, config.manifest_val, "validation", cases))


def make_test_dataset(config):
    """Load the saved test manifest without reshuffling or writing it."""
    validate_saved_splits(config)
    return _dataset_for_manifest(config, config.manifest_test, "test")
