"""Segmentation metrics; HD95 is computed only when surface distances are available."""
import numpy as np


def segmentation_metrics(prediction, target, spacing=(1.,1.,1.)):
    """Foreground metrics plus explicit class/region results.

    Values are computed per volume. ``dice``/``iou``/``hd95`` are macro means
    over foreground classes 1..3; the background is excluded. HD95 units follow
    ``spacing`` (the default is voxel units).
    """
    pred=np.asarray(prediction); truth=np.asarray(target); results={"dice":[],"iou":[],"hd95":[]}
    if pred.shape!=truth.shape: raise ValueError("Prediction and target shapes must match")
    per_class={}; sensitivity=[]; specificity=[]
    for cls in range(1,4):
        a,b=pred==cls,truth==cls; inter=np.logical_and(a,b).sum(); denom=a.sum()+b.sum()
        dice=float((2*inter/denom) if denom else 1.)
        union=np.logical_or(a,b).sum(); iou=float(inter/union if union else 1.)
        results["dice"].append(dice); results["iou"].append(iou)
        tp=inter; fn=b.sum()-tp; fp=a.sum()-tp; tn=b.size-tp-fn-fp
        sens=float(tp/(tp+fn)) if tp+fn else 1.; spec=float(tn/(tn+fp)) if tn+fp else 1.
        sensitivity.append(sens); specificity.append(spec)
        if not a.any() or not b.any(): results["hd95"].append(float("nan"))
        else:
            try:
                from scipy.ndimage import binary_erosion, distance_transform_edt
                sa=a^binary_erosion(a); sb=b^binary_erosion(b)
                da=distance_transform_edt(~sa,sampling=spacing)[sb]; db=distance_transform_edt(~sb,sampling=spacing)[sa]
                results["hd95"].append(float(np.percentile(np.concatenate((da,db)),95)))
            except ImportError: results["hd95"].append(float("nan"))
        per_class[str(cls)]={"dice":dice,"iou":iou,"sensitivity":sens,"specificity":spec,
                             "hd95":results["hd95"][-1]}
    summary={k:float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan") for k,v in results.items()}
    summary.update({"per_class":per_class,"sensitivity":float(np.mean(sensitivity)),
                    "specificity":float(np.mean(specificity)),
                    "regions":{}})
    for name, p, t in (("whole_tumor",pred>0,truth>0),
                       ("tumor_core",np.isin(pred,(1,3)),np.isin(truth,(1,3))),
                       ("enhancing_tumor",pred==3,truth==3)):
        inter=np.logical_and(p,t).sum(); den=p.sum()+t.sum(); union=np.logical_or(p,t).sum()
        fn=t.sum()-inter; fp=p.sum()-inter; tn=t.size-inter-fn-fp
        summary["regions"][name]={"dice":float(2*inter/den) if den else 1.,
            "iou":float(inter/union) if union else 1.,
            "sensitivity":float(inter/t.sum()) if t.sum() else 1.,
            "specificity":float(tn/(tn+fp)) if tn+fp else 1.}
    return summary
