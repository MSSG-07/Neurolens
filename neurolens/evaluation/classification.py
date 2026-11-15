"""Classification evaluation."""
import torch
from neurolens.utils.metrics import classification_metrics


def evaluate_classification(model,loader,device):
    model.eval(); ys=[]; preds=[]; probs=[]
    with torch.inference_mode():
        for batch in loader:
            out=model(batch["image2d"].to(device).float(),batch["image3d"].to(device).float())
            p=out["classification_logits"].softmax(1); ys.extend(batch["class_label"].tolist()); preds.extend(p.argmax(1).cpu().tolist()); probs.extend(p.cpu().tolist())
    return classification_metrics(ys,preds,probs)
