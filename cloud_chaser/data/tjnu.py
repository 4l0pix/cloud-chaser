from __future__ import annotations

from pathlib import Path

import cv2
from torch.utils.data import Dataset

from cloud_chaser.data.gcd import IMAGE_EXTENSIONS


class UnlabeledCloudDataset(Dataset):
    """Unlabeled image dataset for SimCLR/MoCo-style pretraining."""

    def __init__(self, root: str | Path, transform) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Unlabeled dataset not found: {self.root}")
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found under {self.root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = cv2.cvtColor(cv2.imread(str(self.paths[index])), cv2.COLOR_BGR2RGB)
        view1 = self.transform(image=image)["image"]
        view2 = self.transform(image=image)["image"]
        return view1, view2
