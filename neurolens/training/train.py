"""Dataset split and training entry point helpers."""
from neurolens.datasets.brats import discover_patients, split_patients, BraTSPairedDataset


def make_datasets(config):
    cases=discover_patients(config.data_root,config.modalities)
    train_ids,val_ids=split_patients([c[0] for c in cases],config.val_fraction,config.seed)
    train_cases=[c for c in cases if c[0] in set(train_ids)]; val_cases=[c for c in cases if c[0] in set(val_ids)]
    args={"modalities":config.modalities,"patch_size":config.patch_size,"slice_axis":config.slice_axis}
    return BraTSPairedDataset(config.data_root,train_cases,**args),BraTSPairedDataset(config.data_root,val_cases,**args)
