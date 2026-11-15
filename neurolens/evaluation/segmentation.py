"""Segmentation metrics; HD95 is computed only when surface distances are available."""
import numpy as np


def segmentation_metrics(prediction, target, spacing=(1.,1.,1.)):
    pred=np.asarray(prediction); truth=np.asarray(target); results={"dice":[],"iou":[],"hd95":[]}
    if pred.shape!=truth.shape: raise ValueError("Prediction and target shapes must match")
    for cls in range(1,4):
        a,b=pred==cls,truth==cls; inter=np.logical_and(a,b).sum(); denom=a.sum()+b.sum()
        results["dice"].append(float((2*inter/denom) if denom else 1.))
        union=np.logical_or(a,b).sum(); results["iou"].append(float(inter/union if union else 1.))
        if not a.any() or not b.any(): results["hd95"].append(float("nan"))
        else:
            try:
                from scipy.ndimage import binary_erosion, distance_transform_edt
                sa=a^binary_erosion(a); sb=b^binary_erosion(b)
                da=distance_transform_edt(~sa,sampling=spacing)[sb]; db=distance_transform_edt(~sb,sampling=spacing)[sa]
                results["hd95"].append(float(np.percentile(np.concatenate((da,db)),95)))
            except ImportError: results["hd95"].append(float("nan"))
    return {k:float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan") for k,v in results.items()}
