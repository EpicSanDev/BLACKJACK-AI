from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_SOURCE_DIR = BASE_DIR / "dataset" / "generated"
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "val"
VAL_SPLIT = 0.2


def _clean_directory(directory: Path) -> None:
    """Remove existing files/sub-folders in the target directory."""
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return

    for item in directory.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink(missing_ok=True)
        else:
            shutil.rmtree(item, ignore_errors=True)


def _paired_label(image_path: Path) -> Path:
    """Return the expected label path for an image file."""
    return image_path.with_suffix('.txt')


def split_dataset(
    source_dir: Path,
    train_dir: Path,
    val_dir: Path,
    val_split: float = VAL_SPLIT,
    seed: Optional[int] = None,
    copy: bool = False,
) -> Tuple[Sequence[Path], Sequence[Path]]:
    """Split a directory of YOLO-format images into train/val folders.

    Parameters
    ----------
    source_dir: Path
        Directory containing synthesized images and matching label files.
    train_dir: Path
        Destination directory that will receive the training subset.
    val_dir: Path
        Destination directory that will receive the validation subset.
    val_split: float
        Ratio of the dataset to allocate to validation (between 0 and 1).
    seed: Optional[int]
        Optional RNG seed for deterministic shuffling.
    copy: bool
        If True, copy files instead of moving them. Moving keeps storage usage
        lower but destroys the source directory contents.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1 (exclusive).")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    images = sorted(source_dir.glob('*.png'))
    if not images:
        raise FileNotFoundError(f"No PNG images found in {source_dir}")

    rng = random.Random(seed)
    rng.shuffle(images)

    split_index = max(1, int(len(images) * (1.0 - val_split)))
    train_images = images[:split_index]
    val_images = images[split_index:]

    _clean_directory(train_dir)
    _clean_directory(val_dir)

    mover = shutil.copy2 if copy else shutil.move

    def _transfer(files: Iterable[Path], destination: Path) -> None:
        for image_path in files:
            label_path = _paired_label(image_path)
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label file for {image_path.name}")
            mover(str(image_path), str(destination / image_path.name))
            mover(str(label_path), str(destination / label_path.name))

    _transfer(train_images, train_dir)
    _transfer(val_images, val_dir)

    return train_images, val_images


def main() -> None:
    print("Splitting dataset into training and validation sets...")
    train_images, val_images = split_dataset(
        DATA_SOURCE_DIR,
        TRAIN_DIR,
        VAL_DIR,
        val_split=VAL_SPLIT,
        seed=42,
        copy=False,
    )
    print(f"Moved {len(train_images)} image/label pairs to {TRAIN_DIR}")
    print(f"Moved {len(val_images)} image/label pairs to {VAL_DIR}")
    print("Splitting complete.")


if __name__ == '__main__':
    main()
