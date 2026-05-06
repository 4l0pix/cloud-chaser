from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training", "Train", "Training"),
    "val": ("val", "valid", "validation", "Val", "Valid", "Validation"),
    "test": ("test", "testing", "Test", "Testing"),
}
CLASS_KEYWORDS = (
    "cumulus",
    "altocumulus",
    "cirrus",
    "clearsky",
    "clear",
    "stratocumulus",
    "cumulonimbus",
    "mixed",
)
FORBIDDEN_CLASSIFIER_ROOTS = ("swimseg", "skyimage", "sky-image", "segmentation", "mask")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int


def _is_forbidden_classifier_root(path: Path) -> bool:
    text = str(path).lower()
    return any(token in text for token in FORBIDDEN_CLASSIFIER_ROOTS)


def _has_images(path: Path) -> bool:
    return path.exists() and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in path.rglob("*"))


def _image_count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _find_split_dir(root: Path, split: str) -> Path | None:
    for name in SPLIT_ALIASES[split]:
        candidate = root / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _class_dirs(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_dir() and _has_images(p))


def _looks_like_class_root(path: Path) -> bool:
    dirs = _class_dirs(path)
    if len(dirs) < 2:
        return False
    names = " ".join(d.name.lower().replace("_", "") for d in dirs)
    keyword_hits = sum(1 for keyword in CLASS_KEYWORDS if keyword in names)
    return keyword_hits >= 2 or len(dirs) >= 5


def _candidate_roots(root: Path) -> list[Path]:
    candidates: list[Path] = []
    if root.exists():
        candidates.append(root)
        candidates.extend(p for p in root.rglob("*") if p.is_dir())
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(p for p in kaggle_input.glob("*") if p.is_dir())
        candidates.extend(p for p in kaggle_input.rglob("*") if p.is_dir())

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def resolve_gcd_root(root: str | Path) -> Path:
    """Resolve GCD roots in common local and Kaggle layouts.

    Accepted layouts:
    - root/train/<class>/*.jpg
    - root/<class>/*.jpg, in which case train/val/test are stratified in memory
    """
    root = Path(root)
    scored: list[tuple[int, int, Path]] = []
    for path in _candidate_roots(root):
        if _is_forbidden_classifier_root(path):
            continue
        train_dir = _find_split_dir(path, "train")
        if train_dir is not None and _class_dirs(train_dir):
            scored.append((_image_count(train_dir), 0, path))
        elif _looks_like_class_root(path):
            scored.append((_image_count(path), 1, path))

    if scored:
        return sorted(scored, key=lambda item: (-item[0], item[1], len(item[2].parts), str(item[2])))[0][2]

    available = []
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for path in sorted(kaggle_input.glob("*")):
            available.append(f"{path} images={_image_count(path)}")
    raise FileNotFoundError(
        "Could not find GCD classification images. Set data.gcd_root to a folder containing "
        "either train/<class>/*.jpg or <class>/*.jpg. Available Kaggle inputs: " + "; ".join(available)
    )


def discover_classes(root: str | Path) -> list[str]:
    root = resolve_gcd_root(root)
    train_root = _find_split_dir(root, "train") or root
    classes = sorted(p.name for p in _class_dirs(train_root))
    if not classes:
        raise RuntimeError(f"No class folders with images found under {train_root}")
    return classes


def _records_for_split(split_dir: Path | None, classes: list[str]) -> list[ImageRecord]:
    if split_dir is None or not split_dir.exists():
        return []
    records: list[ImageRecord] = []
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    available = {p.name.lower(): p for p in split_dir.iterdir() if p.is_dir()}
    for class_name in classes:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            class_dir = available.get(class_name.lower())
        if class_dir is None or not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(ImageRecord(path=path, label=class_to_idx[class_name]))
    return records


def _split_records(
    records: list[ImageRecord],
    split: str,
    val_fraction: float,
    seed: int,
) -> list[ImageRecord]:
    labels = [r.label for r in records]
    indices = list(range(len(records)))
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=val_fraction * 2,
        random_state=seed,
        stratify=labels,
    )
    holdout_labels = [records[i].label for i in holdout_idx]
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=0.5,
        random_state=seed,
        stratify=holdout_labels,
    )
    selected = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    return [records[i] for i in selected]


def build_gcd_records(
    root: str | Path,
    split: str,
    classes: list[str] | None = None,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[str]]:
    root = resolve_gcd_root(root)
    discovered = discover_classes(root)
    classes = classes or discovered

    train_dir = _find_split_dir(root, "train")
    probe_root = train_dir or root
    probe_records = _records_for_split(probe_root, classes)
    if not probe_records:
        print(f"Configured GCD classes {classes} did not match folders under {probe_root}.")
        print(f"Using discovered classes instead: {discovered}")
        classes = discovered

    if train_dir is None:
        all_records = _records_for_split(root, classes)
        if not all_records:
            raise RuntimeError(f"No GCD images found under class folders in {root}")
        return _split_records(all_records, split, val_fraction, seed), classes

    if split == "test":
        records = _records_for_split(_find_split_dir(root, "test"), classes)
        if records:
            return records, classes
        train_records = _records_for_split(train_dir, classes)
        return _split_records(train_records, "test", val_fraction, seed), classes

    val_dir = _find_split_dir(root, "val")
    if split == "val" and val_dir is not None:
        records = _records_for_split(val_dir, classes)
        if records:
            return records, classes

    train_records = _records_for_split(train_dir, classes)
    if not train_records:
        raise RuntimeError(f"No GCD train images found under {train_dir}. Check data.gcd_root.")
    labels = [r.label for r in train_records]
    indices = list(range(len(train_records)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    selected = train_idx if split == "train" else val_idx
    return [train_records[i] for i in selected], classes


class GCDDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform=None,
        classes: list[str] | None = None,
        val_fraction: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.records, self.classes = build_gcd_records(root, split, classes, val_fraction, seed)
        self.transform = transform
        print(f"GCD {split}: {len(self.records)} images, classes={self.classes}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = cv2.cvtColor(cv2.imread(str(record.path)), cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = Image.open(record.path).convert("RGB")
        return image, record.label
