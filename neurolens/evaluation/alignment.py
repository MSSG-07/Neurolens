"""Paired cross-dimensional cosine similarity."""
import torch


def alignment_similarity(model,loader,device):
    model.eval(); values=[]
    with torch.inference_mode():
        for batch in loader:
            out=model(batch["image2d"].to(device).float(),batch["image3d"].to(device).float())
            values.extend(torch.nn.functional.cosine_similarity(out["embedding2d"],out["embedding3d"],dim=1).cpu().tolist())
    return {"paired_cosine_similarity":sum(values)/max(1,len(values)),"n_pairs":len(values)}
