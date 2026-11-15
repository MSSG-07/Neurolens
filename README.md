# NeuroLens

NeuroLens is a research prototype for **shared cross-dimensional representation learning** from BraTS MRI. A 2D slice sampled from a patient is paired with that same patient's four-modality 3D patch. Separate CNN pathways encode each dimensionality; projection heads map their features to a normalized shared 256-D space. A symmetric in-batch InfoNCE objective aligns matching patient rows and treats other patients as negatives. The 3D pathway also predicts voxel labels, and the 2D pathway predicts a coarse patient-level tumor-presence label.

The prototype does not include pretrained weights or trained results. All dataset experiments are **pending** until executed with real BraTS data. The CPU smoke test uses temporary random tensors strictly for software verification.

## Setup

Use an environment with PyTorch already installed and compatible with the available CUDA runtime. NeuroLens deliberately does not pin or install PyTorch, CUDA, NumPy, Pandas, or Requests. Install only missing optional dependencies after checking your runtime:

```bash
python -m pip install -r requirements.txt
```

NIfTI loading requires NiBabel. YAML configuration requires PyYAML. Training requires PyTorch and NumPy. Optional evaluation/plots use scikit-learn, SciPy, matplotlib, pandas, and Pillow.

## Connect BraTS 2023 GLI

Provide the dataset locally or mount it in Colab. The default expects one directory per patient, containing exactly one NIfTI file for each configured suffix and one `*seg.nii*` file:

```
data/BraTS2023_GLI/
  BraTS-GLI-00000-000/
    BraTS-GLI-00000-000-t1n.nii.gz
    BraTS-GLI-00000-000-t1c.nii.gz
    BraTS-GLI-00000-000-t2w.nii.gz
    BraTS-GLI-00000-000-t2f.nii.gz
    BraTS-GLI-00000-000-seg.nii.gz
```

Set `data_root` and `modalities` in `config.yaml` if names differ. The loader rejects incomplete cases and shape mismatches with patient-specific errors. It reads volumes with NiBabel, maps BraTS label `4` to contiguous class `3` (label set 0/1/2/3), replaces non-finite intensities, performs per-modality nonzero z-scoring, center-crops/pads to `patch_size`, and samples a nonempty slice from the same preprocessed case. The current simple crop may omit peripheral anatomy for data larger than the patch; choose a size appropriate for your dataset.

The deterministic split happens at patient level before sampling, so slices from one patient cannot leak across train/validation. At least two complete patients are needed. Each dataset index represents one patient and samples a slice on access. Batch size must be at least two for aligned conditions B/C so InfoNCE has valid negatives; use `drop_last` behavior configured by the trainer. Small GPU batches and accumulation are recommended for a Colab T4. If memory is insufficient, reduce patch dimensions and/or base channels.

BRISC is a separate optional 2D classification loader. It reads `root/class_name/image` folders and is never paired with BraTS. It is not wired into the paired BraTS trainer because those examples have no originating BraTS volume.

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

Open `notebooks/NeuroLens_BraTS_Colab.ipynb`. Set the dataset path after mounting Drive or uploading the data. The notebook preserves the existing Colab Torch stack and launches the same CLI; it does not download or fabricate patient data.

## Limitations

- This is a minimal research prototype, not a clinically validated system.
- No BraTS data or trained checkpoints are included; all experiment results are pending.
- The classification task is binary tumor presence derived from the segmentation mask; it is not a validated patient diagnosis.
- Patches use deterministic center crop/pad without registration, augmentation, or lesion-aware sampling.
- No distributed training, scheduler, early stopping, full-volume sliding-window inference, or formal calibration is included.
- HD95 implementation is per class then averaged, and empty-class handling reports NaN.
- The independent condition A shares one process/model wrapper but sets cross-dimensional loss weights to zero; report this as a no-alignment multitask baseline, not as a fully isolated single-path experiment.
