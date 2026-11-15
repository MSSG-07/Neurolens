"""Metric helpers, with explicit undefined-case handling."""
import numpy as np


def classification_metrics(y_true, y_pred, probabilities=None):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0)
    result = {"accuracy":float(accuracy_score(y_true,y_pred)),"precision":float(precision),"recall":float(recall),"f1":float(f1)}
    if probabilities is not None:
        try:
            scores = np.asarray(probabilities)
            if scores.ndim == 2 and scores.shape[1] == 2:
                scores = scores[:, 1]
                result["roc_auc"] = float(roc_auc_score(y_true, scores))
            elif scores.ndim == 2:
                result["roc_auc"] = float(roc_auc_score(y_true, scores, multi_class="ovr"))
            else:
                result["roc_auc"] = float(roc_auc_score(y_true, scores))
        except ValueError: result["roc_auc"] = float("nan")
    return result
