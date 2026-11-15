# NeuroLens experiment report

**Status: experiments not run.** This workspace has no PyTorch/NumPy runtime and no BraTS 2023 GLI data. No model observations or performance values are available yet. Do not interpret the pending rows as zero scores.

## Run status

| Condition | Configuration | Training observations | Validation observations | Status |
|---|---|---|---|---|
| A | Classification + segmentation; alignment losses disabled | Pending | Pending | Not run |
| B | A + symmetric cross-dimensional InfoNCE | Pending | Pending | Not run |
| C | B + shared-embedding consistency loss | Pending | Pending | Not run |

## Evaluation criteria

| Task | Criterion | Reporting method |
|---|---|---|
| Classification | Accuracy | Fraction correct on held-out patients |
| Classification | Precision, recall, F1 | Weighted average; zero-division handled as zero |
| Classification | ROC-AUC | One-vs-rest for multiclass; undefined cases reported NaN |
| Segmentation | Dice, IoU | Per foreground class (NCR/NET, ED, ET), then mean |
| Segmentation | HD95 | Per foreground class in voxel units (1 mm default spacing), then mean; NaN if class absent or SciPy unavailable |
| Alignment | Paired cosine similarity | Mean normalized 2D/3D embedding cosine on held-out patient pairs |
| Optimization | Training/validation losses | Epoch means for total, classification, segmentation, contrastive, and consistency objectives |

Classification currently means binary tumor presence derived from whether any non-background voxel appears in the volume. It is not a tumor-subtype diagnosis. For fair condition comparisons, use the same patient split, seeds, preprocessing, training budget, and model capacity; record the config and checkpoint for every run.

## Observations to fill from actual runs

| Condition | Best validation epoch | Val classification loss | Val segmentation loss | Val contrastive loss | Val consistency loss | Accuracy | Precision | Recall | F1 | ROC-AUC | Dice | IoU | HD95 | Paired cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | Pending | Pending | Pending | N/A | N/A | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| B | Pending | Pending | Pending | Pending | N/A | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| C | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

## Loss charts

After training, `runs/neurolens/training.csv` contains epoch-level observations. Generate the loss curve with:

```python
from neurolens.evaluation.visualization import plot_training_curves
plot_training_curves("runs/neurolens/training.csv", "runs/neurolens/loss_curves.png")
```

The chart is intentionally not pre-populated: drawing a performance curve before real training would fabricate observations.
