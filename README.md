# Cloud Chaser

Cloud Chaser is now a U-Net-only cloud segmentation and cloud-type classification research pipeline. It compares six U-Net variants trained on SWIMSEG/SkyImage masks, converts connected cloud regions into RGB crops, then classifies each crop into one of the seven GCD cloud categories with a contrastive self-supervised ResNet50 classifier.

```text
image
  -> U-Net cloud segmentation variant A-F
  -> threshold cloud probability map
  -> connected cloud components + boxes
  -> RGB bounding-box crop from the original image
  -> GCD cloud-type classifier
  -> mask/box/class overlay and cascade metrics
```

The segmentation mask is used for localization and visualization only. The classifier receives the normal RGB crop from the original image, not a blacked-out mask crop.

## Repository Layout

```text
kaggle/       Kaggle notebook and Kaggle-specific workflow files
local-exec/   Local Python package, CLI scripts, configs, and training code
docs/         Documentation, notes, papers, and architecture plans
venv/         Suggested local virtual environment location
requirements.txt
data/         Local datasets, ignored by git
models/       Local exported model artifacts, ignored by git except docs
README.md
results/      Local training/evaluation/inference outputs, ignored by git except docs
```

## Architecture

### 1. U-Net Segmentation Experiments

Each segmenter predicts a binary cloud probability map:

```text
input:  RGB image, resized to 224 x 224
output: 1-channel cloud logit map
```

At inference time:

```text
sigmoid(logits)
threshold at unet.threshold
remove connected components smaller than unet.min_area
extract boxes from surviving components
confidence = mean cloud probability inside component
```

The six research variants are:

```text
Baseline A: compact U-Net, features=[32,64,128,256]
Baseline B: Medium-style U-Net, features=[64,128,256,512]
Model C:    dilated encoder U-Net
Model D:    dilated + ASPP encoder U-Net
Model E:    dilated + ASPP encoder with bicubic decoder
Model F:    improved U-Net with DS skip path + Im-CSAM attention
```

### 2. CSSL Classification

The classifier follows the contrastive self-supervised learning paper in `docs/papers/`: ResNet50 is first pretrained with MoCo-style InfoNCE contrastive learning, then fine-tuned with a fully connected classification head on the seven GCD classes:

```text
1_cumulus
2_altocumulus
3_cirrus
4_clearsky
5_stratocumulus
6_cumulonimbus
7_mixed
```

The classifier input is the RGB crop inside the U-Net component bounding box.

### 3. Cascade Evaluation

GCD has image-level class labels but no cloud masks, so final validation uses image-level cascade metrics:

- **segmentation gate accuracy**: cloudy classes should produce at least one U-Net cloud component; clear-sky images should produce none.
- **classifier accuracy given component**: classification accuracy only among images where a cloud component was produced.
- **cascade accuracy**: end-to-end success after both segmentation gate and classification.

The report outputs are:

```text
results/reports/gcd_val_unet_cascade_bar.png
results/reports/gcd_val_unet_cascade_overlay_samples.jpg
results/reports/gcd_val_unet_cascade_metrics.json
results/reports/unet_ablation/unet_ablation_summary.csv
results/reports/unet_ablation/unet_ablation_summary.png
```

On Kaggle, the equivalent paths are under:

```text
/kaggle/working/reports
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
python train.py unet --config configs/default.yaml
python train.py classifier-ssl --config configs/default.yaml
python train.py classifier --config configs/default.yaml
```

Run the full six-pipeline research comparison:

```bash
python scripts/unet_ablation_report.py --config configs/default.yaml --output-dir ../results/reports/unet_ablation
```

Evaluate:

```bash
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
results/unet/
results/classifier_ssl/
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

The epoch value is a total target, not an additive number. For example, if the U-Net has reached epoch 40 and `epochs: 80`, the next run continues toward epoch 80.
