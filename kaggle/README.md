# Kaggle

This folder contains Kaggle-specific files.

Use `cloud_chaser_kaggle.ipynb` in Kaggle. The notebook materializes the Python project inside `/kaggle/working`, restores saved checkpoints from an attached output dataset when available, resumes training, and writes results to simple Kaggle paths:

```text
/kaggle/working/runs
/kaggle/working/reports
/kaggle/working/checkpoints
/kaggle/working/artifacts
/kaggle/working/prediction.jpg
```

The validation report cell now runs an experimental backend comparison across:

```text
yolo
unet
hybrid
```

It writes:

```text
/kaggle/working/reports/gcd_backend_experiment_comparison.png
/kaggle/working/reports/gcd_backend_experiment_summary.csv
/kaggle/working/reports/gcd_val_yolo_cascade_*.png|jpg|json
/kaggle/working/reports/gcd_val_unet_cascade_*.png|jpg|json
/kaggle/working/reports/gcd_val_hybrid_cascade_*.png|jpg|json
```

The notebook should be treated as the Kaggle entrypoint; the local code lives in `local-exec/`.
