"""Train NeuroLens on a real BraTS directory."""
import argparse
from neurolens.config import load_config
from neurolens.utils.reproducibility import seed_everything
from neurolens.models.multitask import NeuroLens
from neurolens.training.trainer import Trainer
from neurolens.training.train import make_datasets


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="config.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--condition",choices=("A","B","C"))
    parser.add_argument("--output-dir")
    args=parser.parse_args(); cfg=load_config(args.config)
    if args.data_root: cfg.data_root=args.data_root
    if args.condition: cfg.alignment_condition=args.condition
    if args.output_dir: cfg.output_dir=args.output_dir
    cfg.validate(); seed_everything(cfg.seed)
    train_data,val_data=make_datasets(cfg)
    model=NeuroLens(len(cfg.modalities),cfg.base_channels,cfg.projection_dim,cfg.num_classes,cfg.classification_classes)
    Trainer(model,cfg).fit(train_data,val_data)


if __name__=="__main__": main()
