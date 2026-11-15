"""Paired training loop with AMP and gradient accumulation."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from neurolens.losses.multitask_loss import MultiTaskLoss
from neurolens.training.checkpoint import save_checkpoint
from neurolens.utils.logging import CSVLogger


class Trainer:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.device=torch.device(config.device if config.device=="cpu" or torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        self.scaler=torch.amp.GradScaler("cuda",enabled=config.mixed_precision and self.device.type=="cuda")
        weights={"classification":config.lambda_classification,"segmentation":config.lambda_segmentation,
                 "contrastive":config.lambda_contrastive if config.alignment_condition in ("B","C") else 0.,
                 "consistency":config.lambda_consistency if config.alignment_condition=="C" else 0.}
        self.loss=MultiTaskLoss(weights,config.temperature)
        self.logger=CSVLogger(Path(config.output_dir)/"training.csv")

    def run_epoch(self, loader, training=True):
        self.model.train(training); totals={}; n=0
        if training: self.optimizer.zero_grad(set_to_none=True)
        for i,batch in enumerate(loader):
            image2d=batch["image2d"].to(self.device,non_blocking=True).float()
            image3d=batch["image3d"].to(self.device,non_blocking=True).float()
            targets={"class_label":batch["class_label"].to(self.device),"segmentation":batch["segmentation"].to(self.device)}
            if training and self.config.alignment_condition in ("B","C") and image2d.shape[0]<2:
                raise ValueError("Aligned training requires batch_size >= 2 for valid in-batch negatives")
            with torch.set_grad_enabled(training), torch.autocast(device_type=self.device.type,enabled=self.scaler.is_enabled()):
                output=self.model(image2d,image3d); loss,parts=self.loss(output,targets)
                scaled=loss/self.config.accumulation_steps
            if training:
                self.scaler.scale(scaled).backward()
                if (i+1)%self.config.accumulation_steps==0 or (i+1)==len(loader):
                    self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad(set_to_none=True)
            for key,value in parts.items(): totals[key]=totals.get(key,0.)+float(value.detach())
            n+=1
        return {k:v/max(1,n) for k,v in totals.items()}

    def fit(self, train_dataset, val_dataset):
        cfg=self.config
        train_loader=DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,
                                pin_memory=self.device.type=="cuda",drop_last=cfg.alignment_condition in ("B","C") and cfg.batch_size>=2)
        val_loader=DataLoader(val_dataset,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=self.device.type=="cuda")
        if cfg.alignment_condition in ("B","C") and cfg.batch_size<2: raise ValueError("Set batch_size >= 2 for InfoNCE negatives")
        for epoch in range(cfg.epochs):
            train=self.run_epoch(train_loader,True); val=self.run_epoch(val_loader,False)
            row={"epoch":epoch+1,**{f"train_{k}":v for k,v in train.items()},**{f"val_{k}":v for k,v in val.items()}}
            self.logger.log(row); print(row)
            save_checkpoint(Path(cfg.output_dir)/"latest.pt",self.model,self.optimizer,self.scaler,epoch+1,cfg.__dict__)
