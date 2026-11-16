# NeuroLens Research Training Pipeline

Primary notebook: [NeuroLens_Complete_Training_Pipeline.ipynb](NeuroLens_Complete_Training_Pipeline.ipynb)

This notebook preserves the repository's NeuroLens paired 2D/3D architecture and adds a configurable comparison with volumetric segmentation baselines, leakage-aware prediction stacking utilities, and convolutional Grad-CAM. BRISC remains an independent 2D audit dataset; it is never paired with BraTS.

## Colab setup and execution

1. Open the primary notebook in Google Colab and mount Google Drive when prompted.
2. Select a CUDA GPU for practical execution. If CUDA is not present, the notebook states that it is using CPU; 3D training can be substantially slower.
3. Run all cells from the top. The one master switch is `RUN_ALL_EXPERIMENTS = True` in the A/B/C orchestration cell. `ARCHITECTURE_PLAN` is a recorded experiment configuration for ResUNet3D, Swin UNETR and 3D U-Net.
4. The notebook validates the existing manifests and patient splits, trains/resumes NeuroLens A/B/C first, then executes the enabled segmentation baselines sequentially. It selects checkpoints on validation data and evaluates each selected checkpoint on the held-out test split.

The full experiment is not expected to finish in one Colab session. NeuroLens and each segmentation baseline write independent atomic recovery checkpoints to Drive. Re-run the notebook from the top after reconnecting; compatible completed work is reused and incomplete runs resume at an epoch boundary. A changed device or configuration is intentionally treated as incompatible with an existing checkpoint. `EXECUTION_PLAN` records which optional components are enabled; `RUN_ALL_EXPERIMENTS` is the single master start control.

The CPU option is explicit in the runtime output and run metadata. Mixed precision is disabled on CPU. The main A/B/C configuration remains 50 epochs per condition, batch size 2, accumulation 4, AdamW learning rate 1e-4, weight decay 1e-5 and seed 42. ResUNet3D and 3D U-Net use 50 epochs, batch size 2 and accumulation 4. Swin UNETR uses batch size 1, accumulation 8 and activation checkpointing to reduce peak memory; its effective batch size matches the convolutional baselines. These budgets are not silently reduced.

## Datasets and scientific safeguards

- BraTS root: `MyDrive/NeuroLens/data/raw/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`
- BRISC root: `MyDrive/NeuroLens/data/raw/BRISC2025/brisc2025`
- Existing BraTS manifests: `MyDrive/NeuroLens/manifests/train.csv`, `validation.csv`, and `test.csv`

The saved BraTS patient-level splits (296/37/38) are not regenerated. Modalities remain ordered `t1n`, `t1c`, `t2w`, `t2f`. Source segmentation value 4 is converted by the existing loader to model label 3 (ET). Train slices remain random; validation/test use the middle of the sorted valid-slice indices for each patient. The same deterministic validation and test samples are reused by every condition and architecture.

BRISC is audited separately. Its manifest labels and task-specific loader are not mixed into the paired BraTS batches. In particular, BRISC images are never treated as positive partners for BraTS volumes.

## Experiments

| Model family | Tasks | Notes |
|---|---|---|
| NeuroLens A | Binary tumor presence + 4-class segmentation | Multitask objective; no alignment terms |
| NeuroLens B | Same | Adds symmetric paired InfoNCE, same-patient slice/volume positives |
| NeuroLens C | Same | Adds embedding consistency to B |
| ResUNet3D | 4-class volumetric segmentation | Residual encoder-decoder with skip connections and GroupNorm |
| Swin UNETR | 4-class volumetric segmentation | Optional MONAI implementation; 3D Swin encoder and U-Net decoder |
| UNet3D | 4-class volumetric segmentation | Conventional 3D convolutional reference |

Segmentation-only baselines do not receive a fabricated classification head. Classification comparison is reported for NeuroLens, while segmentation and resource comparisons include all completed models. Inference and validation metrics are patient-wise: foreground class Dice/IoU are macro-averaged within each patient, then averaged across patients. Whole tumor is labels 1|2|3, tumor core is 1|3, enhancing tumor is 3. HD95 uses unit voxel spacing in patch coordinates and is reported in voxels, not millimetres.

The experiment registry records model configuration, parameter counts, status, selected epoch, validation/test summaries, checkpoint, runtime and peak GPU memory when available. A model failure is recorded without deleting successful experiments.

## Meta-learning and prediction caches

The repository provides atomic prediction cache helpers that enforce patient order and provenance, a probability-averaging baseline, logistic regression/random forest stacking, optional XGBoost/LightGBM candidates, and validation-selected segmentation probability fusion.

Classification meta-learning prefers patient-level out-of-fold predictions from the training split when a provenance-compatible cache is present. If no OOF cache exists, the notebook creates deterministic, train-only in-sample predictions from each selected checkpoint and marks the resulting overfit risk in its results. Candidate selection (best individual model, mean/weighted probability fusion, logistic regression, random forest and optional XGBoost/LightGBM) uses validation labels only. The test manifest and labels are not opened until selection is persisted. Segmentation probabilities are cached on validation and simple fusion weights are selected by patient-macro foreground Dice. The frozen choices are evaluated once on the same held-out test cohort.

## Explainability

`neurolens.evaluation.explainability.GradCAM` supports 2D/3D convolutional feature maps. The notebook creates same-patient MRI, ground-truth mask and heatmap overlays for available checkpoints and reports activation ratios and thresholded overlap. These are localization summaries; Grad-CAM alone does not establish causal faithfulness. For segmentation probability fusion, the notebook displays the validation-weighted average of constituent model Grad-CAM maps and explicitly does not call it a joint gradient. A conventional tabular classification stack has no spatial Grad-CAM; its attribution is left unsupported. A Swin decoder-layer Grad-CAM is not attention rollout. Transformer attention rollout is not implemented.

## Outputs

```text
MyDrive/NeuroLens/experiments/
  condition_A/                         # likewise B and C
    latest.pt
    best_validation.pt
    training_state.pt
    training.csv
    config.json
    runtime_config.yaml
    run_metadata.json
    evaluation/validation_results.json
    evaluation/test_results.json
    evaluation/per_case/*.npz
  architectures/
    resunet/
    swin_unetr/
    unet3d/
  predictions/classification/
  predictions/segmentation/
  meta_learning/classification/
  meta_learning/segmentation/
    validation_weight_selection.json
    selected_fusion_test_results.json
    per_case/*.npz
  explainability/
  comparative_analysis/
    architecture_comparison.csv
    architecture_comparison.json
    classification_comparison.csv
    segmentation_comparison.csv
    ensemble_comparison.csv
    ensemble_comparison.json
    comparison.csv
    comparison.json
    final_research_report.md
  experiment_registry.json
  experiment_status.json
```

Missing metrics are left missing; the notebook does not invent scores or statistical significance. Test metrics are only used for final comparison after validation checkpoint selection.

## Dependencies and limitations

The notebook uses existing `requirements.txt` and preserves its PyTorch runtime. MONAI 1.5.2 is installed only when Swin UNETR is enabled and MONAI is absent. Optional XGBoost/LightGBM candidates are skipped when their packages are unavailable. Swin UNETR memory use depends on the Colab GPU; activation checkpointing and micro-batching are configured, but successful training is not assumed. If it fails, the registry records the error and other architectures continue.

The repository's saved BraTS preprocessing crops to 64³ and does not preserve physical-affine spacing in the model-facing patch. Therefore, current HD95 is voxel-space only. Per-case baseline outputs are summarized in JSON; the full-resolution original NIfTI reconstruction/export is not implemented in this extension.

The notebook has not been run through the real-data training workflow as part of these code changes. Static validation and locally available tests are reported separately; do not interpret notebook construction as completed model experiments.
