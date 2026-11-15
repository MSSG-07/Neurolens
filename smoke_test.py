"""CPU software smoke checks using temporary tensors only; no patient results."""
from neurolens.utils.dependencies import check_modules

_, missing, broken = check_modules(("torch", "numpy"))
if missing or broken:
    details = [f"missing packages: {', '.join(missing)}"] if missing else []
    details += [f"{name} import failed: {error}" for name, error in broken]
    raise SystemExit("Smoke test not run; " + "; ".join(details))

import torch
from neurolens.models.multitask import NeuroLens
from neurolens.losses.multitask_loss import MultiTaskLoss


def main():
    torch.manual_seed(3)
    # Compact dimensions keep this check feasible on a CPU.
    model=NeuroLens(in_channels=4,base_channels=4,projection_dim=16,classes=4,classification_classes=2)
    x2=torch.randn(2,4,32,32); x3=torch.randn(2,4,32,32,32)
    target={"class_label":torch.tensor([0,1]),"segmentation":torch.randint(0,4,(2,32,32,32))}
    out=model(x2,x3); loss,parts=MultiTaskLoss()(out,target); loss.backward()
    assert out["embedding2d"].shape==(2,16) and out["embedding3d"].shape==(2,16)
    assert out["segmentation_logits"].shape==(2,4,32,32,32)
    assert torch.isfinite(loss).item() and all(torch.isfinite(v).item() for v in parts.values())
    print("CPU smoke test passed: paired forward, all losses, and backward; synthetic tensors only.")


if __name__=="__main__": main()
