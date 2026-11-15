"""Configurable classification, Dice+CE segmentation, alignment and consistency losses."""
import torch
from torch import nn
from torch.nn import functional as F
from .contrastive import SymmetricInfoNCE


class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None, temperature=.1):
        super().__init__()
        self.weights = {"classification":1., "segmentation":1., "contrastive":.1, "consistency":.05}
        if weights: self.weights.update(weights)
        self.info_nce = SymmetricInfoNCE(temperature)

    def forward(self, output, targets):
        class_loss = F.cross_entropy(output["classification_logits"], targets["class_label"].long())
        logits, truth = output["segmentation_logits"], targets["segmentation"].long()
        ce = F.cross_entropy(logits, truth)
        probs = logits.softmax(1)
        onehot = F.one_hot(truth, logits.shape[1]).movedim(-1,1).to(probs.dtype)
        dims = (0,2,3,4)
        dice = 1 - ((2*(probs*onehot).sum(dims)+1e-5)/(probs.sum(dims)+onehot.sum(dims)+1e-5)).mean()
        seg_loss = ce + dice
        # Both projection outputs share a configured dimension, unlike raw encoder features.
        consistency = 1 - F.cosine_similarity(output["embedding2d"], output["embedding3d"], dim=1).mean()
        # A final validation singleton has no valid negatives; omit only that term.
        contrastive = (self.info_nce(output["embedding2d"], output["embedding3d"])
                       if self.weights["contrastive"] and logits.shape[0] >= 2 else logits.new_zeros(()))
        terms = {"classification":class_loss, "segmentation":seg_loss, "contrastive":contrastive, "consistency":consistency}
        total = sum(self.weights[k]*v for k,v in terms.items())
        return total, {**terms, "total":total}
