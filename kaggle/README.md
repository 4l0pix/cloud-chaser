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

The notebook should be treated as the Kaggle entrypoint; the local code lives in `local-exec/`.
