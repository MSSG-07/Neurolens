# NeuroLens Three-Condition Colab Study

Primary notebook: [`NeuroLens_Complete_Training_Pipeline.ipynb`](NeuroLens_Complete_Training_Pipeline.ipynb)

This notebook runs the repository's existing NeuroLens architecture through the planned BraTS A/B/C ablation study. It keeps BRISC as an independent 2D dataset audit and does not create BRISC-to-BraTS pairs.

## Run in Colab

1. Upload/open `NeuroLens_Complete_Training_Pipeline.ipynb` in Colab.
2. Select a CUDA GPU runtime and use Runtime → Run all.
3. Approve the Google Drive mount prompt.
4. The notebook verifies the repository, dataset paths, saved split counts, and disjoint patient IDs. It then runs Conditions A, B, and C sequentially through the single `RUN_ALL_CONDITIONS = True` control.

The master cell refuses to train without CUDA. Training is 50 epochs per condition (150 total) with batch size 2, accumulation 4, AdamW at 1e-4 and weight decay 1e-5, mixed precision, and seed 42. It does not add a scheduler, early stopping, gradient clipping, or architecture changes. No one-session runtime guarantee is made; the run is designed to continue across Colab disconnects. After reconnecting, run the notebook from the top again; completed epochs, conditions, validations, and test reports are detected and skipped.

## Required Drive data

- BraTS: `MyDrive/NeuroLens/data/raw/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`
- BRISC: `MyDrive/NeuroLens/data/raw/BRISC2025/brisc2025`
- Saved manifests: `MyDrive/NeuroLens/manifests/train.csv`, `validation.csv`, and `test.csv`

The notebook never regenerates the saved splits. Validation and test use a deterministic slice: the middle entry of the sorted valid-slice indices for each patient. Training retains random valid-slice sampling. This makes the same validation/test patient view apply to A, B, and C while preserving same-patient slice/volume correspondence.

## Conditions

| Condition | Classification | Segmentation | Contrastive | Consistency |
|---|---:|---:|---:|---:|
| A | 1.0 | 1.0 | 0.0 | 0.0 |
| B | 1.0 | 1.0 | 0.1 | 0.0 |
| C | 1.0 | 1.0 | 0.1 | 0.05 |

Each condition resets the same seed and initializes a fresh model, AdamW optimizer, scaler, and DataLoader generator. Conditions B/C use `drop_last=True` to guarantee two distinct patients per InfoNCE batch. The test manifest remains unopened until all three best checkpoints are selected using validation total loss and final validation reports are written.

## Outputs

```text
MyDrive/NeuroLens/experiments/
  experiment_status.json
  condition_A/                  # same layout for B and C
    latest.pt
    best_validation.pt
    training_state.pt           # canonical atomic recovery checkpoint
    training.csv
    config.json
    runtime_config.yaml
    run_metadata.json
    evaluation/
      validation_results.json
      test_results.json
      test_confusion_matrix.png
      per_case/*.npz
      qualitative/*.png
  comparative_analysis/
    comparison.csv
    comparison.json
    paired_patient_differences.json
    comparison_plots/
    confusion_matrices/
    qualitative_results/
    final_research_report.md
```

Per-case predictions are saved as cropped 64³ patch-space NPZ arrays, not restored full-resolution NIfTI files. HD95 uses the existing utility's unit voxel spacing; undefined values are represented as JSON `null`. Classification precision/recall/F1 are support-weighted and computed globally over patients. Segmentation Dice/IoU are foreground class metrics macro-averaged within each patient, then averaged across patients. Empty-empty Dice/IoU is 1, consistent with the repository utility.

## Resume and safety

`training_state.pt` contains model, optimizer, AMP scaler, epoch, full history, best-validation state, elapsed time, configuration, Python/NumPy/Torch/CUDA RNG state, and the DataLoader generator state. Resume validates the saved config and condition metadata, then restores the next epoch. Checkpoint writes use the repository's atomic checkpoint helper. `latest.pt` and `best_validation.pt` are mirrors; missing mirrors can be repaired from canonical state when possible. Corrupt or incompatible state stops with an error instead of silently starting over. CSV history is atomically rebuilt from checkpoint history to avoid duplicate rows.

The checked-in `config.yaml` has blank data paths. The notebook fills the existing Config fields from the verified Drive locations at runtime and writes the resolved per-condition configuration into each experiment folder; it does not edit `config.yaml` or saved manifests.

