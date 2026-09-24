# Landslide segmentation on aerial imagery

The repository contains the segmentation models and training pipeline
described in Davide Sarti's bachelor's thesis, [Segmentazione automatica di
frane su immagini aeree](https://amslaurea.unibo.it/id/eprint/36917/), completed at the University of Bologna in 2024/2025.
It also includes shared raster utilities and code related to colleagues'
super-resolution and synthetic cloud generation experiments.

The segmentation task predicts a binary landslide mask from
pre-event and post-event aerial imagery.

## Setup

Run commands from the repository root. Python 3.11--3.13 is recommended. A
CUDA-enabled PyTorch installation is useful for training, but CPU execution is
supported for inspection and small checks.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dataset, thesis checkpoints and generated artifacts are not portable
project inputs. They must be supplied locally before training or evaluation.

## Dataset layout

The loader expects four municipality directories under `Comuni/`:

```text
Comuni/<municipality>/
├── Agea_2020_2m.tif          # pre-event aerial RGB + NIR
├── Cgr_2023_2m.tif           # post-event aerial RGB + NIR; valid-area mask
├── Frane_V1_clipped.tif      # landslide reference mask
├── Agea_NDVI_2020_2m.tif     # optional derived feature
├── Cgr_NDVI_2023_2m.tif      # optional derived feature
├── Change_NDVI_ortho_2m.tif  # optional derived feature
├── Change_NDVI_S2_2m.tif     # optional derived feature
├── Slope_2m.tif              # optional derived feature
├── Sentinel2_pre_2m.tif      # used by super-resolution, not segmentation
└── Sentinel2_post_2m.tif     # used by super-resolution, not segmentation
```

Supported municipality names are `Brisighella`, `Casola-Valsenio`,
`Modigliana` and `Predappio`.

The files have these roles:

| File | Description |
|---|---|---|
| `Agea_2020_2m.tif` | Pre-event aerial imagery, with four RGB+NIR bands. |
| `Cgr_2023_2m.tif` | Post-event aerial imagery, with four RGB+NIR bands. |
| `Frane_V1_clipped.tif` | Reference landslide inventory. Values greater than zero are converted to the binary target mask. |
| `Agea_NDVI_2020_2m.tif` | NDVI derived from the pre-event AGEA imagery. |
| `Cgr_NDVI_2023_2m.tif` | NDVI derived from the post-event Cgr imagery. |
| `Change_NDVI_ortho_2m.tif` | NDVI change map computed from the two orthophotos. |
| `Change_NDVI_S2_2m.tif` | NDVI change map derived from Sentinel-2 data. |
| `Slope_2m.tif` | Slope derived from elevation data, expressed in degrees. |

The loaders check that input and target stacks have equal height and width, but
they do not reproject rasters or verify their CRS, transform or pixel size.
Prepare all products on the same grid before running the pipeline.

### Channels and normalization

Raster values are converted to `float32`, nodata values become `NaN`, and the
following ranges are normalized to `[0, 1]`:

| Product | Expected range |
|---|---:|
| `Agea` and `Cgr` | `[0, 255]` |
| `Slope` | `[0, 90]` |
| `NDVI` and change maps | `[-1, 1]` for NDVI, `[-2, 2]` for change |
| Landslide raster | `[0, 8]`, then thresholded at `> 0` |

Without `--include_slope_ndvi`, the segmentation input is normally the eight
RGB+NIR channels from the pre-event and post-event aerial products. With the
flag, files whose names contain `slope` or `ndvi` are also considered. The
exact channel count depends on the filenames and band counts present in the
municipality directory.

## Segmentation training

Use module execution from the repository root so package-relative imports are
resolved correctly:

```bash
python -m Segmentation.train --help
python -m Segmentation.train --model unet --loss bce
python -m Segmentation.train --model attunet --loss bce_dice
python -m Segmentation.train --model swin --loss bce
python -m Segmentation.train --model unet --loss bce_dice --include_slope_ndvi
```

Available options are:

| Option | Values / default | Meaning |
|---|---|---|
| `--model` | `unet`, `attunet`, `swin` / `unet` | Architecture |
| `--loss` | `bce`, `dice`, `sqdice`, `bce_dice` / `bce` | Training loss |
| `--include_slope_ndvi` | flag | Include files named `slope` or `ndvi` |
| `--weights-dir` | `Segmentation/weights` | Model checkpoints |
| `--metrics-dir` | `Segmentation/artifacts/metrics` | JSON metric histories |
| `--plots-dir` | `Segmentation/artifacts/plots` | Training plots |

The training configuration uses `256x256` patches, generates 4000 random
patches per dataset epoch, trains with batch size 24 and evaluates with batch
size 8. The fixed geographic split is:

```text
training:   Brisighella, Modigliana, Predappio
validation: Casola-Valsenio
```

Patches are sampled on the fly. The implemented geometric and radiometric
augmentation helper supports flips, 90-degree rotations, brightness and
contrast changes.

The optimizer is AdamW with learning rate `5e-4`. Weight decay is `1e-4` for
U-Net and Attention U-Net and `5e-2` for Swin U-Net. A
`ReduceLROnPlateau` scheduler halves the learning rate after five stagnant
validation-loss epochs. Training stops after 15 epochs without IoU
improvement, but only after epoch 30, with a maximum of 120 epochs.

`bce_dice` is calculated as `0.7 * BCE + 0.3 * Dice`. The best checkpoint is
selected by validation IoU and contains only the model `state_dict`.

### Models

- **U-Net**: convolutional encoder-decoder with skip connections.
- **Attention U-Net**: U-Net with attention gates on the skip connections.
- **Swin U-Net**: hierarchical Swin Transformer encoder-decoder. Its checked-in
	configuration expects eight input channels and a `256x256` image, so extended
	channel stacks may require a corresponding model/configuration change.

## Evaluation

Evaluate a checkpoint on a municipality with:

```bash
python -m Segmentation.evaluate \
	--weights Segmentation/weights/unet_bce_model.pth \
	--comune Casola-Valsenio \
	--save-visuals
```

Useful options:

```bash
python -m Segmentation.evaluate --help
python -m Segmentation.evaluate --weights path/to/model.pth --variable-threshold
python -m Segmentation.evaluate --weights path/to/model.pth --save-visuals \
	--visuals-dir Segmentation/artifacts/qualitative/my_run
```

The evaluator reports loss, IoU, Dice, overall accuracy and a normalized
confusion matrix. Its default CLI threshold is set by FIXED_THRESHOLD; with
`--variable-threshold`, thresholds from `0.20` to `0.80` are scanned in steps
of `0.02` and `threshold_metrics.png` is written to the plots directory.

For compatibility with the existing checkpoints, the evaluator infers the
architecture from the checkpoint path (`unet`, `attunet` or `swin`) and uses
`BCEDiceLoss(bce_weight=0.7)` for evaluation. Therefore, evaluating a model
trained with another loss is not a perfectly matched loss comparison. Napari
is imported when the evaluator starts; use the full requirements environment
even when saving Matplotlib visuals headlessly.

## Stored outputs

Training writes files using the prefix `<model>_<loss>` and adds
`_slope_ndvi` when requested:

```text
Segmentation/weights/<prefix>_model.pth
Segmentation/artifacts/metrics/<prefix>_metrics.json
Segmentation/artifacts/plots/<prefix>_losses.png
Segmentation/artifacts/plots/<prefix>_oa.png
Segmentation/artifacts/plots/<prefix>_iou.png
```

Evaluation visualizations are written under
`Segmentation/artifacts/qualitative/` when `--save-visuals` is used. Without
that flag, the evaluator opens an interactive napari viewer.

## Common tools

The root-level utilities are shared by the segmentation and super-resolution
experiments:

```bash
python dimensions_check.py Comuni --output raster_coherence.csv
python view_dataset.py
```

- `data_utils.py` loads and normalizes GeoTIFFs, creates the valid-area mask,
	samples aligned patches, applies optional augmentation and exposes the
	segmentation/super-resolution dataset classes.
- `dimensions_check.py` compares the dimensions of the 2 m aerial products and
	the 10 m Sentinel products for every municipality and writes a CSV report. (not relevant for this task)
- `view_dataset.py` opens an interactive napari viewer for complete rasters or
	random patches. It is interactive and does not expose a non-interactive CLI.
- `view_utils.py` provides napari layers, histograms and product-specific
	colormaps.

The viewer tools require a graphical session. For remote or headless runs,
prefer `Segmentation.evaluate --save-visuals`.

## Experimental context and reported results

The following tables preserve the reported segmentation results from the
original experiments. They are historical results rather than independently
reproduced benchmarks. The first table compares the BCE weight `lambda` in
`lambda * BCE + (1 - lambda) * Dice`; the threshold is selected during
evaluation for each model and loss-weight configuration. Bold values identify
the best IoU reported for each architecture.

### Loss-weight and architecture comparison


| Architecture | λ | Mean Loss | Accuracy | IoU | Threshold |
|---|---|---|---|---|---|
| **U-Net** | 1.0 | 0.0812 | 0.9717 | 0.5574 | 0.48 |
| **U-Net** | **0.7** | 0.2256 | 0.9726 | **0.5665** | 0.58 |
| **U-Net** | 0.5 | 0.2320 | 0.9717 | 0.5516 | 0.60 |
| **U-Net** | 0.3 | 0.2403 | 0.9715 | 0.5501 | 0.58 |
| **Attention U-Net** | 1.0 | 0.1846 | 0.9720 | 0.5590 | 0.42 |
| **Attention U-Net** | **0.7** | 0.1779 | 0.9722 | **0.5622** | 0.44 |
| **Attention U-Net** | 0.5 | 0.1807 | 0.9719 | 0.5567 | 0.52 |
| **Attention U-Net** | 0.3 | 0.1986 | 0.9713 | 0.5480 | 0.60 |
| **Swin U-Net** | **1.0** | 0.2031 | 0.9708 | **0.5442** | 0.38 |
| **Swin U-Net** | 0.7 | 0.1893 | 0.9696 | 0.5271 | 0.44 |
| **Swin U-Net** | 0.5 | 0.1953 | 0.9684 | 0.5173 | 0.42 |
| **Swin U-Net** | 0.3 | 0.2199 | 0.9686 | 0.5120 | 0.48 |


| Model | Optimal λ | Parameters | TP | FP | TN | FN | Accuracy | IoU |
|---|---|---|---|---|---|---|---|---|
| **U-Net** | 0.7 | 31,046,401 | 0.0358 | 0.0135 | 0.9368 | 0.0139 | 0.9726 | 0.5665 |
| Attention U-Net | 0.7 | 31,396,005 | 0.0357 | 0.0138 | 0.9365 | 0.0140 | 0.9722 | 0.5622 |
| Swin U-Net | 1.0 (BCE only) | 31,745,364 | 0.0349 | 0.0143 | 0.9359 | 0.0149 | 0.9708 | 0.5442 |

## Project structure

```text
Segmentation/
├── train.py                 # training entry point
├── evaluate.py              # checkpoint evaluation and visualization
├── losses.py                # BCE, Dice and combined losses
├── models/                  # U-Net, Attention U-Net and Swin U-Net
├── scripts/                 # comparison plots and qualitative exporters
├── weights/                 # checked-in/example checkpoints
└── artifacts/               # metrics, plots and qualitative outputs

data_utils.py                # shared raster loading and patch datasets
dimensions_check.py          # spatial-dimension consistency report
view_dataset.py              # interactive raster/patch viewer
view_utils.py                # napari plotting helpers
Cloud_Generation/            # synthetic cloud experiments
Super_Resolution/            # super-resolution experiments
```
