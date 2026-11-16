"""Metric helpers, with explicit undefined-case handling."""
import numpy as np


def classification_metrics(y_true, y_pred, probabilities=None):
    from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)
    observed = set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist())
    labels = [0, 1] if observed <= {0, 1} else sorted(observed)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(y_true,y_pred,labels=labels,average=None,zero_division=0)
    result = {"accuracy":float(accuracy_score(y_true,y_pred)),"precision":float(precision),"recall":float(recall),"f1":float(f1),
              "macro_precision":float(macro_p),"macro_recall":float(macro_r),"macro_f1":float(macro_f1),
              "per_class":{str(label):{"precision":float(p_cls[i]),"recall":float(r_cls[i]),"f1":float(f_cls[i]),"support":int(support[i])}
                           for i,label in enumerate(labels)},
              "confusion_matrix":confusion_matrix(y_true,y_pred,labels=labels).tolist(),
              "confusion_matrix_labels":labels}
    if probabilities is not None:
        try:
            scores = np.asarray(probabilities)
            if scores.ndim == 2 and scores.shape[1] == 2:
                scores = scores[:, 1]
                result["roc_auc"] = float(roc_auc_score(y_true, scores))
                result["pr_auc"] = float(average_precision_score(y_true, scores))
            elif scores.ndim == 2:
                result["roc_auc"] = float(roc_auc_score(y_true, scores, multi_class="ovr"))
            else:
                result["roc_auc"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            result["roc_auc"] = float("nan")
            result["pr_auc"] = float("nan")
    return result
