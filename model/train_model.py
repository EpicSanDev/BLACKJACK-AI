from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from prepare_dataset import split_dataset
from utils import select_best_device

DEFAULT_CARD_DIR = BASE_DIR / "dataset" / "png"
DEFAULT_BACKGROUND_DIR = BASE_DIR / "dataset" / "backgrounds"
GENERATED_DIR = BASE_DIR / "dataset" / "generated"
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "val"
DATASET_CONFIG = BASE_DIR / "dataset.yaml"
DEFAULT_WEIGHTS = BASE_DIR / "yolov8n.pt"
DEFAULT_PROJECT = BASE_DIR / "runs" / "detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic dataset generation and YOLOv8 training pipeline for Blackjack card detection"
    )
    parser.add_argument("--card-dir", type=Path, default=DEFAULT_CARD_DIR, help="Path to transparent card PNGs")
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BACKGROUND_DIR, help="Path holding table background images")
    parser.add_argument("--generated-dir", type=Path, default=GENERATED_DIR, help="Directory where synthetic samples are written")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR, help="Output directory for YOLO training images")
    parser.add_argument("--val-dir", type=Path, default=VAL_DIR, help="Output directory for YOLO validation images")
    parser.add_argument("--dataset-config", type=Path, default=DATASET_CONFIG, help="YOLO dataset YAML file")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Base YOLO weights to fine-tune")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Directory that will contain YOLO runs")
    parser.add_argument("--name", type=str, default="blackjack_detector_max", help="Run name under the project directory")
    parser.add_argument("--num-images", type=int, default=2500, help="Number of synthetic images to generate")
    parser.add_argument("--img-width", type=int, default=832, help="Synthetic image width")
    parser.add_argument("--img-height", type=int, default=640, help="Synthetic image height")
    parser.add_argument("--min-cards", type=int, default=2, help="Minimum number of cards per synthetic frame")
    parser.add_argument("--max-cards", type=int, default=5, help="Maximum number of cards per synthetic frame")
    parser.add_argument("--epochs", type=int, default=120, help="Number of YOLO training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=896, help="YOLO input size (images resized to this square resolution)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Proportion of data reserved for validation")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="Number of dataloader worker processes")
    parser.add_argument("--device", type=str, default=None, help="Force a specific device string understood by Ultralytics")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    parser.add_argument("--no-regenerate", dest="regenerate", action="store_false", help="Skip dataset regeneration and reuse existing train/val folders")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer to use for training")
    parser.add_argument("--lr0", type=float, default=1e-3, help="Initial learning rate passed to Ultralytics")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience in epochs")
    parser.add_argument("--cache-images", action="store_true", help="Cache images in memory to accelerate training")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision even if available")
    parser.set_defaults(regenerate=True, amp=True)
    return parser.parse_args()


def load_card_images(card_dir: Path) -> Dict[str, np.ndarray]:
    if not card_dir.exists():
        raise FileNotFoundError(f"Card directory does not exist: {card_dir}")

    card_images: Dict[str, np.ndarray] = {}
    for image_path in sorted(card_dir.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.shape[2] != 4:
            raise ValueError(f"Card image {image_path.name} must include an alpha channel.")
        card_images[image_path.stem] = image

    if not card_images:
        raise RuntimeError(f"No valid card PNGs were found in {card_dir}")
    return card_images


def load_backgrounds(background_dir: Path, target_size: Tuple[int, int]) -> List[np.ndarray]:
    if not background_dir.exists():
        return []

    backgrounds: List[np.ndarray] = []
    for bg_path in sorted(background_dir.glob("*")):
        if bg_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        backgrounds.append(resized)

    return backgrounds


def _gradient_background(width: int, height: int, rng: random.Random) -> np.ndarray:
    top = np.array([rng.randint(10, 80) for _ in range(3)], dtype=np.float32)
    bottom = np.array([rng.randint(40, 120) for _ in range(3)], dtype=np.float32)
    gradient = np.zeros((height, width, 3), dtype=np.float32)
    for row in range(height):
        alpha = row / max(1, height - 1)
        gradient[row, :] = (1 - alpha) * top + alpha * bottom

    np_rng = np.random.default_rng(rng.getrandbits(32))
    gradient += np_rng.normal(0, 4.0, size=gradient.shape)
    np.clip(gradient, 0, 255, out=gradient)
    return gradient.astype(np.uint8)


def _alpha_blend(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape[:2]
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    inv_alpha = 1.0 - alpha
    dst_region = dst[top : top + h, left : left + w].astype(np.float32)
    blended = alpha * src[:, :, :3] + inv_alpha * dst_region
    dst[top : top + h, left : left + w] = np.clip(blended, 0, 255).astype(np.uint8)


def _transform_card(card: np.ndarray, angle: float, scale: float) -> np.ndarray:
    scaled = cv2.resize(card, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = scaled.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]
    rotated = cv2.warpAffine(
        scaled,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated


def synthesise_scene(
    width: int,
    height: int,
    card_names: Sequence[str],
    card_images: Dict[str, np.ndarray],
    backgrounds: List[np.ndarray],
    card_lookup: Dict[str, int],
    rng: random.Random,
    min_cards: int,
    max_cards: int,
) -> Tuple[np.ndarray, List[str]]:
    canvas = (
        rng.choice(backgrounds).copy()
        if backgrounds
        else _gradient_background(width, height, rng)
    )

    num_cards = rng.randint(min_cards, max_cards)
    labels: List[str] = []
    attempts = 0

    while len(labels) < num_cards and attempts < num_cards * 4:
        attempts += 1
        card_name = rng.choice(card_names)
        card = card_images[card_name]
        angle = rng.uniform(-18.0, 18.0)
        scale = rng.uniform(0.72, 1.08)
        transformed = _transform_card(card, angle, scale)
        h, w = transformed.shape[:2]

        if h >= height or w >= width:
            scale_factor = min((height - 10) / max(h, 1), (width - 10) / max(w, 1))
            if scale_factor <= 0:
                continue
            transformed = cv2.resize(transformed, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            h, w = transformed.shape[:2]

        max_top = height - h - 1
        max_left = width - w - 1
        if max_top <= 5 or max_left <= 5:
            continue

        top = rng.randint(5, max_top)
        left = rng.randint(5, max_left)

        _alpha_blend(canvas, transformed, top, left)

        cx = (left + w / 2) / width
        cy = (top + h / 2) / height
        bw = w / width
        bh = h / height
        labels.append(f"{card_lookup[card_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if rng.random() < 0.35:
        blur_kernel = rng.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (blur_kernel, blur_kernel), sigmaX=0.0)

    if rng.random() < 0.4:
        brightness = rng.uniform(0.9, 1.1)
        contrast = rng.uniform(0.9, 1.1)
        canvas = np.clip((canvas.astype(np.float32) * contrast) + ((brightness - 1.0) * 40), 0, 255).astype(np.uint8)

    return canvas, labels


def clean_generation_directories(*directories: Path) -> None:
    for directory in directories:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def remove_yolo_caches(dataset_dir: Path) -> None:
    for cache_path in dataset_dir.glob("*.cache"):
        cache_path.unlink(missing_ok=True)


def generate_dataset(
    args: argparse.Namespace,
    card_names: Sequence[str],
    card_images: Dict[str, np.ndarray],
    backgrounds: List[np.ndarray],
    card_lookup: Dict[str, int],
) -> None:
    rng = random.Random(args.seed)
    args.generated_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.num_images):
        image, labels = synthesise_scene(
            args.img_width,
            args.img_height,
            card_names,
            card_images,
            backgrounds,
            card_lookup,
            rng,
            args.min_cards,
            args.max_cards,
        )
        image_path = args.generated_dir / f"image_{index:05d}.png"
        label_path = args.generated_dir / f"image_{index:05d}.txt"
        cv2.imwrite(str(image_path), image)
        label_path.write_text("\n".join(labels))

        if (index + 1) % 50 == 0 or index + 1 == args.num_images:
            print(f"Generated {index + 1}/{args.num_images} synthetic samples", end="\r", flush=True)

    print(f"\nSynthetic dataset generation complete: {args.num_images} images saved to {args.generated_dir}")


def train(args: argparse.Namespace) -> None:
    print("Loading assets...")
    card_images = load_card_images(args.card_dir)
    card_names: Sequence[str] = tuple(card_images.keys())
    backgrounds = load_backgrounds(args.background_dir, (args.img_width, args.img_height))
    card_lookup = {name: idx for idx, name in enumerate(card_names)}

    if args.regenerate:
        print("Regenerating dataset and resetting splits...")
        clean_generation_directories(args.generated_dir, args.train_dir, args.val_dir)
        remove_yolo_caches(args.generated_dir.parent)
        generate_dataset(args, card_names, card_images, backgrounds, card_lookup)
        split_dataset(
            args.generated_dir,
            args.train_dir,
            args.val_dir,
            val_split=args.val_split,
            seed=args.seed,
            copy=False,
        )
        print(f"Train set prepared at {args.train_dir}, validation at {args.val_dir}")
    else:
        print("Skipping dataset regeneration as requested.")

    remove_yolo_caches(args.train_dir.parent)

    device = select_best_device(args.device)
    print(f"Training on device: {device}")
    print(f"Using {args.workers} dataloader workers")

    model = YOLO(str(args.weights))
    results = model.train(
        data=str(args.dataset_config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        cache=args.cache_images,
        amp=args.amp,
        seed=args.seed,
        pretrained=True,
    )

    print("Training complete. Best weights stored at:")
    print(results.best)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
