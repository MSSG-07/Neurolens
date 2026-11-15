"""Evaluate a NeuroLens checkpoint on the patient-held-out BraTS split."""
import argparse, json
import torch
from torch.utils.data import DataLoader
from neurolens.config import load_config
from neurolens.models.multitask import NeuroLens
from neurolens.training.train import make_datasets
from neurolens.training.checkpoint import load_checkpoint
from neurolens.evaluation.classification import evaluate_classification
from neurolens.evaluation.alignment import alignment_similarity
from neurolens.evaluation.segmentation import segmentation_metrics


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--checkpoint",required=True); a=p.parse_args()
    cfg=load_config(a.config); _,data=make_datasets(cfg); model=NeuroLens(len(cfg.modalities),cfg.base_channels,cfg.projection_dim,cfg.num_classes,cfg.classification_classes)
    device=torch.device(cfg.device if cfg.device=="cpu" or torch.cuda.is_available() else "cpu"); load_checkpoint(a.checkpoint,model,map_location=device); model.to(device)
    loader=DataLoader(data,batch_size=cfg.batch_size,num_workers=cfg.num_workers)
    result={"classification":evaluate_classification(model,loader,device),"alignment":alignment_similarity(model,loader,device)}
    model.eval(); rows=[]
    with torch.inference_mode():
        for batch in loader:
            out=model(batch["image2d"].to(device).float(),batch["image3d"].to(device).float())
            pred=out["segmentation_logits"].argmax(1).cpu().numpy()
            for i in range(len(pred)): rows.append(segmentation_metrics(pred[i],batch["segmentation"][i].numpy()))
    result["segmentation"]={k:sum(r[k] for r in rows if r[k]==r[k])/max(1,sum(r[k]==r[k] for r in rows)) for k in ("dice","iou","hd95")}
    print(json.dumps(result,indent=2,allow_nan=True))


if __name__=="__main__": main()
