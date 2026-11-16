"""Atomic checkpoints with optimizer/scaler and reproducibility state."""
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, scaler, epoch, config, extra_state=None):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    checkpoint = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
                  "epoch":epoch,"config":config,"torch_rng":torch.get_rng_state()}
    if extra_state is not None:
        checkpoint["extra_state"] = extra_state
    torch.save(checkpoint,tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu"):
    state=torch.load(path,map_location=map_location,weights_only=False); model.load_state_dict(state["model"])
    if optimizer is not None: optimizer.load_state_dict(state["optimizer"])
    if scaler is not None: scaler.load_state_dict(state["scaler"])
    if "torch_rng" in state: torch.set_rng_state(state["torch_rng"])
    return int(state["epoch"])
