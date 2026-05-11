# Cloud Chaser

Cloud Chaser is a hybrid cloud detection and cloud-type classification pipeline. It detects cloud regions with a YOLO-Seg primary detector and a U-Net fallback detector, then classifies the detected RGB crop into one of the seven GCD cloud categories.

The current best pipeline is:

```text
image
  -> YOLO-Seg cloud detector
  -> if YOLO finds no cloud, U-Net fallback detector
  -> RGB bounding-box crop from the original image
  -> GCD cloud-type classifier
  -> mask/box/class overlay and cascade metrics
```

The detector mask is used for localization and visualization only. The classifier receives the normal RGB crop from the original image, not a blacked-out mask crop.

## Repository Layout

```text
kaggle/       Kaggle notebook and Kaggle-specific workflow files
local-exec/   Local Python package, CLI scripts, configs, and training code
docs/         Documentation, notes, and papers
venv/         Suggested local virtual environment location
requirements.txt
data/         Local datasets, ignored by git
models/       Local pretrained/downloaded weights, ignored by git except docs
README.md
results/      Local training/evaluation/inference outputs, ignored by git except docs
```

## Architecture

### 1. Hybrid Detection

The detector stage has two backends:

- **YOLO-Seg primary detector**: fast instance segmentation, trained from `yolo11s-seg.pt` on SWIMSEG cloud masks converted to YOLO polygon labels.
- **U-Net fallback detector**: dense semantic segmentation trained on the same SWIMSEG binary masks. It runs when YOLO returns no detections.

YOLO returns masks, boxes, and detector confidences directly. U-Net returns a cloud probability map; the pipeline thresholds it, removes small components, extracts connected components, and converts those components into masks, boxes, and confidence scores.

### 2. Classification

The classifier is a PyTorch CNN, currently ResNet50 by default, trained on the seven GCD classes:

```text
1_cumulus
2_altocumulus
3_cirrus
4_clearsky
5_stratocumulus
6_cumulonimbus
7_mixed
```

The classifier input is the RGB crop inside the detector bounding box. The detector mask is not applied to classifier pixels.

### 3. Cascade Evaluation

GCD has image-level class labels but no cloud masks, so final validation uses image-level cascade metrics:

- **detector image accuracy**: cloudy classes should produce at least one detection; clear-sky images should produce none.
- **classifier accuracy given detection**: classification accuracy only among images where a cloud was detected.
- **cascade accuracy**: end-to-end success after both detection and classification.

The report outputs are:

```text
results/reports/gcd_val_cascade_bar.png
results/reports/gcd_val_cascade_overlay_samples.jpg
results/reports/gcd_val_cascade_metrics.json
```

On Kaggle, the equivalent paths are:

```text
/kaggle/working/reports/gcd_val_cascade_bar.png
/kaggle/working/reports/gcd_val_cascade_overlay_samples.jpg
/kaggle/working/reports/gcd_val_cascade_metrics.json
```

## Local Quick Start

Create a virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e local-exec
```

Expected local data layout:

```text
data/swimseg-2/**              # SWIMSEG/SkyImage image-mask pairs
data/GCD/train/<class>/*.jpg
data/GCD/test/<class>/*.jpg
```

Run local training from `local-exec/`:

```bash
cd local-exec
python -m scripts.prepare_swimseg_yolo --config configs/default.yaml
python train.py detector --config configs/default.yaml
python train.py unet --config configs/default.yaml
python train.py classifier --config configs/default.yaml
```

Evaluate:

```bash
python train.py eval-detector --config configs/default.yaml
python train.py eval-unet --config configs/default.yaml
python train.py eval-classifier --config configs/default.yaml
python scripts/gcd_visual_report.py --config configs/default.yaml --output-dir ../results/reports
```

Run inference:

```bash
python inference.py \
  --config configs/default.yaml \
  --image ../data/example.jpg \
  --output ../results/prediction.jpg
```

## Kaggle Workflow

Use:

```text
kaggle/cloud_chaser_kaggle.ipynb
```

The notebook writes its own working copy to Kaggle, restores checkpoints from the attached `latest-output` dataset when present, resumes training from `last.pt` or `best.pt`, and writes simple outputs to:

```text
/kaggle/working/runs
/kaggle/working/reports
/kaggle/working/checkpoints
/kaggle/working/artifacts
/kaggle/working/prediction.jpg
```

## Checkpoints

Local checkpoints are written under:

```text
results/detector/weights/
results/unet/
results/classifier/
```

The training code resumes in this order:

```text
last.pt -> best.pt -> start fresh
```

For longer experiments, increase the target total epoch counts in:

```text
local-exec/configs/default.yaml
```

The epoch value is a total target, not an additive number. For example, if the classifier has reached epoch 40 and `epochs: 80`, the next run continues toward epoch 80.
