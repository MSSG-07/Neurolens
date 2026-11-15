"""BRISC loader placeholder pending inspection of its Drive manifests and labels."""
from torch.utils.data import Dataset


class BRISCDataset(Dataset):
    """Separate BRISC dataset entry point; preprocessing is deliberately gated.

    The supplied Drive tree contains multiple task directories and manifest files.
    Until their README, manifest schemas, labels, and image organization are inspected,
    guessing a folder-to-label mapping would risk corrupting the experiment.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "BRISC preprocessing is pending read-only inspection of README.md, "
            "manifest.csv/json, checksums, and task directories. Run "
            "inspect_datasets.py or the Colab setup notebook first."
        )
