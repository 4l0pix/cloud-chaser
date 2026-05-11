# Local Execution

This folder contains the local Python implementation:

```text
cloud_chaser/   Python package
configs/        local YAML configs
scripts/        dataset conversion and reporting scripts
train.py        training/evaluation CLI
inference.py    two-stage inference CLI
export.py       export CLI
pyproject.toml  editable install metadata
```

Run commands from this directory so relative paths in `configs/default.yaml` resolve to the repo-level `data/`, `models/`, and `results/` folders.

```bash
cd local-exec
python train.py detector --config configs/default.yaml
python train.py unet --config configs/default.yaml
python train.py classifier --config configs/default.yaml
```
