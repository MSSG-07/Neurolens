# NeuroLens Complete Training Pipeline Notebook

Notebook: [`NeuroLens_Complete_Training_Pipeline.ipynb`](NeuroLens_Complete_Training_Pipeline.ipynb)

This Colab notebook documents and runs the repository's existing NeuroLens pipeline for a controlled BraTS Condition A baseline. It includes Drive setup, source inspection, read-only BRISC and BraTS audits, saved-split leakage checks, preprocessing visualization, real DataLoader inspection, the existing architecture and losses, opt-in training with resumable Drive checkpoints, held-out test evaluation, qualitative outputs, and a generated experiment report.

## Execution order

1. Open the notebook in Google Colab and select a GPU runtime (T4 or better recommended). Setup, source inspection, manifest checks, and visual inspection can run without starting training; dataset decoding requires the project dependencies and mounted Drive data.
2. Run setup and data-audit cells from the top. The notebook uses the saved BraTS manifests at `MyDrive/NeuroLens/manifests/{train,validation,test}.csv` and does not regenerate them.
3. Review the split, sample, preprocessing, and model/loss descriptions.
4. Training is disabled by default. Set `START_TRAINING = True` in the dedicated Condition A cell and execute that cell explicitly when ready. The output directory is `MyDrive/NeuroLens/runs/baseline_A`.
5. For recovery, set `RESUME = True` and execute the same training cell. It restores `latest.pt`, optimizer/scaler state, history, best validation loss, and Python/NumPy/CUDA RNG state when available.
6. After training/model selection, set `RUN_FINAL_EVALUATION = True` in the evaluation cell and execute it. It evaluates the best-validation checkpoint on the saved test manifest and writes metrics and per-case `.npz` prediction artifacts.

## Data and compute

The expected Drive roots are:

- BraTS 2023 GLI: `MyDrive/NeuroLens/data/raw/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`
- BRISC 2025: `MyDrive/NeuroLens/data/raw/BRISC2025/brisc2025`

The notebook checks these paths and stops with a clear message if they are absent. BRISC folder variants with `(1)` are resolved by the repository's `BRISCPathResolver`. BRISC remains independent 2D data; it is not used to construct cross-dimensional positives. NeuroLens's current model and loss consume same-patient BraTS 2D/3D pairs.

The checked-in `config.yaml` has empty dataset-root and manifest fields. The notebook explicitly fills the existing `Config` object's `brats_root`, `data_root`, and saved manifest fields after verifying the Drive files; it does not edit `config.yaml` or regenerate manifests. When training starts, that resolved runtime configuration is recorded as `config.json` and `runtime_config.yaml` in the experiment directory.

The checked-in config defaults to batch size 2, 64-cubed patches, 50 epochs, AdamW at `1e-4` with `1e-5` weight decay, and mixed precision when CUDA is available. Four-channel 3D volumes are memory intensive; the notebook refuses to start training without CUDA by default. Lowering batch size or patch size changes the baseline and should be recorded as a separate experiment. Gradient accumulation is configured to 4 steps.

## Output layout

```text
MyDrive/NeuroLens/runs/baseline_A/
  config.json
  runtime_config.yaml
  run_metadata.json
  training.csv
  latest.pt
  best_validation.pt
  training_state.pt
  plots/
  evaluation/results.json
  evaluation/confusion_matrix.png
  evaluation/per_case/*.npz
  experiment_report.md
```

The `.npz` files contain the model's cropped 64-cubed prediction and ground truth in the dataset's `(D,H,W)=(Z,Y,X)` orientation, plus classification values and patient ID. They are patch-space artifacts, not restored full-resolution NIfTI volumes. No source dataset or saved manifest is modified.

## Method notes

- Condition A is the existing baseline: classification and segmentation losses are active; contrastive and consistency weights are set to zero by `Trainer` for condition A.
- Classification is binary tumour presence, generated from non-background voxels in the loaded cropped segmentation. It is not BRISC's four-class tumour subtype prediction.
- Segmentation labels follow the existing loader conversion from BraTS source `{0,1,2,4}` to contiguous `{0,1,2,3}`. The loss averages soft Dice across all four classes and adds multiclass cross entropy.
- The model's paired 2D slice and 3D crop come from the same patient. Condition A does not optimize a contrastive term; no cross-dataset positive pairs are created.
- The repository trainer has no scheduler, early stopping, or built-in resume/best-checkpoint selection. The notebook uses its `Trainer.run_epoch`, `save_checkpoint`, and `load_checkpoint` APIs and adds notebook-level recovery/reporting without changing repository source.
- Test data is not loaded until the final evaluation section, after best checkpoint selection using validation loss.

