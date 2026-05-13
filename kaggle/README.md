# Kaggle

This folder contains Kaggle-specific files.

Use `cloud_chaser_kaggle.ipynb` in Kaggle. The notebook materializes the Python project inside `/kaggle/working`, restores saved U-Net/classifier checkpoints from an attached output dataset when available, resumes training, and writes results to simple Kaggle paths:

```text
/kaggle/working/runs
/kaggle/working/reports
/kaggle/working/checkpoints
/kaggle/working/artifacts
/kaggle/working/prediction.jpg
```

The validation report writes:

```text
/kaggle/working/reports/gcd_val_unet_cascade_bar.png
/kaggle/working/reports/gcd_val_unet_cascade_overlay_samples.jpg
/kaggle/working/reports/gcd_val_unet_cascade_metrics.json
/kaggle/working/reports/unet_ablation/unet_ablation_summary.csv
/kaggle/working/reports/unet_ablation/unet_ablation_summary.png
```

The notebook is now the experimental entrypoint for comparing all six U-Net segmenters before the GCD classifier. The local code lives in `local-exec/`.
