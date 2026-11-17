"""Paired training loop with AMP and gradient accumulation."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from neurolens.losses.multitask_loss import MultiTaskLoss
from neurolens.training.checkpoint import save_checkpoint
from neurolens.utils.logging import CSVLogger
from neurolens.utils.metrics import classification_metrics


class Trainer:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.device=torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.xla_model=None
        if self.device.type == "xla":
            try:
                import torch_xla.runtime as xr
                import torch_xla.core.xla_model as xm
            except ImportError as exc:
                raise RuntimeError("XLA was requested but PyTorch/XLA is unavailable in this runtime") from exc
            if xr.device_type() != "TPU":
                raise RuntimeError(f"XLA device was requested, but PyTorch/XLA reports {xr.device_type()!r}, not TPU")
            self.xla_model=xm
        elif self.device.type not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported training device: {self.device}")
        self.model.to(self.device)
        self.optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        self.scaler=torch.amp.GradScaler("cuda",enabled=config.mixed_precision and self.device.type=="cuda")
        weights={"classification":config.lambda_classification,"segmentation":config.lambda_segmentation,
                 "contrastive":config.lambda_contrastive if config.alignment_condition in ("B","C") else 0.,
                 "consistency":config.lambda_consistency if config.alignment_condition=="C" else 0.}
        self.loss=MultiTaskLoss(weights,config.temperature)
        self.logger=CSVLogger(Path(config.output_dir)/"training.csv")

    def run_epoch(self, loader, training=True, collect_metrics=False):
        """Run one epoch; optionally aggregate deterministic evaluation metrics.

        Metric collection is available only in evaluation mode and happens in
        the same forward pass as validation loss, avoiding a second NIfTI load.
        Classification scores are global across patient predictions. Segmentation
        Dice/IoU are computed per patient (foreground classes 1..3, macro within
        patient) and then averaged across patients. Both-empty regions score 1,
        matching ``segmentation_metrics``; HD95 is intentionally not computed
        per epoch because it is an expensive final-evaluation metric.
        """
        if collect_metrics and training:
            raise ValueError("collect_metrics=True requires training=False")
        self.model.train(training); totals={}; n=0
        metric_truth, metric_prediction = [], []
        patient_dice, patient_iou = [], []
        if training: self.optimizer.zero_grad(set_to_none=True)
        for i,batch in enumerate(loader):
            image2d=batch["image2d"].to(self.device,non_blocking=True).float()
            image3d=batch["image3d"].to(self.device,non_blocking=True).float()
            targets={"class_label":batch["class_label"].to(self.device),"segmentation":batch["segmentation"].to(self.device)}
            if training and self.config.alignment_condition in ("B","C") and image2d.shape[0]<2:
                raise ValueError("Aligned training requires batch_size >= 2 for valid in-batch negatives")
            if self.device.type == "xla":
                autocast_context=torch.autocast(device_type="xla",dtype=torch.bfloat16,enabled=self.config.mixed_precision)
            else:
                autocast_context=torch.autocast(device_type=self.device.type,enabled=self.scaler.is_enabled())
            with torch.set_grad_enabled(training), autocast_context:
                output=self.model(image2d,image3d); loss,parts=self.loss(output,targets)
                scaled=loss/self.config.accumulation_steps
            batch_size = int(image2d.shape[0])
            if collect_metrics:
                probabilities = output["classification_logits"].softmax(1).float().detach().cpu().numpy()
                metric_truth.extend(targets["class_label"].detach().cpu().tolist())
                metric_prediction.extend(probabilities.argmax(1).tolist())
                segmentation = output["segmentation_logits"].argmax(1).detach().cpu().numpy()
                truth = targets["segmentation"].detach().cpu().numpy()
                for prediction, target in zip(segmentation, truth):
                    class_dice, class_iou = [], []
                    for class_id in (1, 2, 3):
                        pred_mask, true_mask = prediction == class_id, target == class_id
                        intersection = np.logical_and(pred_mask, true_mask).sum()
                        denominator = pred_mask.sum() + true_mask.sum()
                        union = np.logical_or(pred_mask, true_mask).sum()
                        class_dice.append(float(2 * intersection / denominator) if denominator else 1.0)
                        class_iou.append(float(intersection / union) if union else 1.0)
                    patient_dice.append(float(np.mean(class_dice)))
                    patient_iou.append(float(np.mean(class_iou)))
            if training:
                self.scaler.scale(scaled).backward()
                if (i+1)%self.config.accumulation_steps==0 or (i+1)==len(loader):
                    if self.device.type == "xla":
                        self.xla_model.optimizer_step(self.optimizer)
                    else:
                        self.scaler.step(self.optimizer); self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            for key,value in parts.items(): totals[key]=totals.get(key,0.)+float(value.detach())*batch_size
            n+=batch_size
        result = {k:v/max(1,n) for k,v in totals.items()}
        if collect_metrics:
            if not metric_truth:
                raise ValueError("Cannot collect validation metrics from an empty DataLoader")
            # ROC-AUC is calculated in final evaluation where undefined values
            # can be reported explicitly; epoch rows contain finite core metrics.
            classification = classification_metrics(metric_truth, metric_prediction)
            # Keep epoch metrics scalar so CSV logging remains well-defined;
            # per-class reports/confusion matrices are emitted in final evaluation.
            result.update({f"classification_{key}": value for key, value in classification.items()
                           if np.isscalar(value)})
            result["segmentation_dice"] = float(np.mean(patient_dice)) if patient_dice else float("nan")
            result["segmentation_iou"] = float(np.mean(patient_iou)) if patient_iou else float("nan")
        return result

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
