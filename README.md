# NeuroLens

NeuroLens is a research prototype for **shared cross-dimensional representation learning** from BraTS MRI. A 2D slice sampled from a patient is paired with that same patient's four-modality 3D patch. Separate CNN pathways encode each dimensionality; projection heads map their features to a normalized shared 256-D space. A symmetric in-batch InfoNCE objective aligns matching patient rows and treats other patients as negatives. The 3D pathway also predicts voxel labels, and the 2D pathway predicts a coarse patient-level tumor-presence label.

The prototype does not include pretrained weights or trained results. All dataset experiments are **pending** until executed with real BraTS data. The CPU smoke test uses temporary random tensors strictly for software verification.

## Setup

Use an environment with PyTorch already installed and compatible with the available CUDA runtime. NeuroLens deliberately does not pin or install PyTorch, CUDA, NumPy, Pandas, or Requests. Install only missing optional dependencies after checking your runtime:

```bash
python -m pip install -r requirements.txt
```

NIfTI loading requires NiBabel. YAML configuration requires PyYAML. Training requires PyTorch and NumPy. Optional evaluation/plots use scikit-learn, SciPy, matplotlib, pandas, and Pillow.

## Dataset paths and saved splits

Set paths in `config.yaml` or override them through `NEUROLENS_BRATS_ROOT`, `NEUROLENS_BRISC_ROOT`, `NEUROLENS_MANIFEST_TRAIN`, `NEUROLENS_MANIFEST_VAL`, `NEUROLENS_MANIFEST_TEST`, and `NEUROLENS_OUTPUT_DIR`. BraTS root should be the directory containing `BraTS-GLI-*` patient folders; BRISC root should be the nested `brisc2025` directory.

NeuroLens loads the existing train, validation, and test patient assignments in manifest order. It does not rewrite manifests or create replacement splits. Configure their original file paths in YAML or environment variables. `manifest_id_column` selects the patient ID column; common case ID names are also recognized.

Inspect the Drive files before choosing BRISC preprocessing:

```bash
python inspect_datasets.py --brats-root "$NEUROLENS_BRATS_ROOT" --brisc-root "$NEUROLENS_BRISC_ROOT" --train-manifest "$NEUROLENS_MANIFEST_TRAIN" --val-manifest "$NEUROLENS_MANIFEST_VAL" --test-manifest "$NEUROLENS_MANIFEST_TEST"
```

The inspector prints BRISC README and manifest metadata, validates listed SHA256 files when present, and samples the directory tree without applying image preprocessing. The BRISC loader is gated and raises an error until its task labels and file organization are confirmed.

## BraTS 2023 GLI layout

Provide the dataset locally or mount it in Colab. The default expects one directory per patient, containing exactly one NIfTI file for each configured suffix and one `*seg.nii*` file:

```
<brats_root>/
  BraTS-GLI-00000-000/
    BraTS-GLI-00000-000-t1n.nii.gz
    BraTS-GLI-00000-000-t1c.nii.gz
    BraTS-GLI-00000-000-t2w.nii.gz
    BraTS-GLI-00000-000-t2f.nii.gz
    BraTS-GLI-00000-000-seg.nii.gz
```

Set `brats_root` and `modalities` in `config.yaml` if names differ. The loader rejects incomplete cases and shape mismatches with patient-specific errors. It reads volumes with NiBabel, maps BraTS label `4` to contiguous class `3` (label set 0/1/2/3), replaces non-finite intensities, performs per-modality nonzero z-scoring, center-crops/pads to `patch_size`, and samples a nonempty slice from the same preprocessed case. The current simple crop may omit peripheral anatomy for data larger than the patch; choose a size appropriate for your dataset.

Patient assignments are loaded from the saved manifests, preserving their order. A manifest patient without a complete case raises an error instead of being silently dropped. Each dataset index represents one patient and samples a slice on access. Batch size must be at least two for aligned conditions B/C so InfoNCE has valid negatives; use `drop_last` behavior configured by the trainer. Small GPU batches and accumulation are recommended for a Colab T4. If memory is insufficient, reduce patch dimensions and/or base channels.

BRISC remains a separate dataset and is never paired with BraTS. Its loader currently raises a clear `NotImplementedError` until the Drive README and manifests have been inspected and the task labels confirmed.

## Run

```bash
python train.py --config config.yaml --data-root /path/to/BraTS2023_GLI --condition C
python evaluate.py --config config.yaml --checkpoint runs/neurolens/latest.pt
python smoke_test.py
```

Conditions are configured as follows:

| Condition | Enabled losses | Status |
|---|---|---|
| A | Classification + Dice/CE segmentation; alignment weights zero | Pending run |
| B | A + symmetric cross-dimensional InfoNCE | Pending run |
| C | B + pooled-feature cosine consistency | Pending run |

Training writes `training.csv` and `latest.pt` under `output_dir`. Checkpoints include model, optimizer, AMP scaler, epoch, config, and Torch RNG state. The run prints epoch losses. Validation currently records multitask losses; use `evaluate.py` for accuracy, weighted precision/recall/F1, ROC-AUC when defined, segmentation Dice/IoU/HD95, and paired embedding cosine similarity. HD95 is NaN if SciPy is absent or a foreground class is missing. Classification labels currently mean whole-volume tumor presence (binary), not a clinical diagnosis or a BRISC tumor subtype.

`neurolens.evaluation.visualization` provides Grad-CAM, slice overlays, embedding scatter plots, and loss curves. These helpers are opt-in and require their plotting dependencies.

## Colab

Open `notebooks/NeuroLens_BraTS_Colab.ipynb`. It mounts Drive, clones or pulls the repository, installs project requirements without reinstalling Torch/NumPy, validates configured dataset/manifests, inspects BRISC metadata, and runs a one-batch real BraTS integration check. It does not start full training.

## Limitations

- This is a minimal research prototype, not a clinically validated system.
- No BraTS data or trained checkpoints are included; all experiment results are pending.
- The classification task is binary tumor presence derived from the segmentation mask; it is not a validated patient diagnosis.
- Patches use deterministic center crop/pad without registration, augmentation, or lesion-aware sampling.
- No distributed training, scheduler, early stopping, full-volume sliding-window inference, or formal calibration is included.
- HD95 implementation is per class then averaged, and empty-class handling reports NaN.
- The independent condition A shares one process/model wrapper but sets cross-dimensional loss weights to zero; report this as a no-alignment multitask baseline, not as a fully isolated single-path experiment.
