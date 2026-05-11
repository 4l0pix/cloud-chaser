# Setup And Run

Use Python 3.10, 3.11, or 3.12. Python 3.14 is not recommended for this stack because PyTorch and several native dependencies may not provide stable wheels.

## Install

From the repository root:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e local-exec
```

For CUDA, install the CUDA-enabled PyTorch wheels before the editable install if your platform needs a specific wheel index.

## Data

Expected local data:

```text
data/swimseg-2/**
data/GCD/train/<class>/*.jpg
data/GCD/test/<class>/*.jpg
```

## Local Training

Run from `local-exec/`:

```bash
cd local-exec
python -m scripts.prepare_swimseg_yolo --config configs/default.yaml
python train.py detector --config configs/default.yaml
python train.py unet --config configs/default.yaml
python train.py classifier --config configs/default.yaml
```

Training resumes from `last.pt` when present, otherwise from `best.pt`, otherwise from scratch.

## Evaluation

```bash
python train.py eval-detector --config configs/default.yaml
python train.py eval-unet --config configs/default.yaml
python train.py eval-classifier --config configs/default.yaml
python scripts/gcd_visual_report.py --config configs/default.yaml --output-dir ../results/reports
```

## Inference

```bash
python inference.py \
  --config configs/default.yaml \
  --image ../data/example.jpg \
  --output ../results/prediction.jpg
```

## Export

```bash
python export.py detector --config configs/default.yaml --format onnx
python export.py classifier --config configs/default.yaml --format torchscript --output ../models/classifier.torchscript
```
