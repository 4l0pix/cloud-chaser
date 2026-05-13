# Local Execution

This folder contains the local Python implementation:

```text
cloud_chaser/   Python package
configs/        local YAML configs
scripts/        dataset conversion and reporting scripts
train.py        training/evaluation CLI
inference.py    U-Net segmentation + classification inference CLI
export.py       export CLI
pyproject.toml  editable install metadata
```

Run commands from this directory so relative paths in `configs/default.yaml` resolve to the repo-level `data/`, `models/`, and `results/` folders.

```bash
cd local-exec
python train.py unet --config configs/default.yaml
python train.py classifier-ssl --config configs/default.yaml
python train.py classifier --config configs/default.yaml
```

Run the six-segmenter research comparison:

```bash
python scripts/unet_ablation_report.py --config configs/default.yaml --output-dir ../results/reports/unet_ablation
```
