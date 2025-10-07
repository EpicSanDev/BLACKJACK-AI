from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import shutil
import sys
import uuid
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
KAGGLE_CARDS_YOLO_DIR = BASE_DIR / "dataset" / "external" / "kaggle_cards_yolo"
DEFAULT_CARD_DISTRIBUTION = BASE_DIR / "dataset" / "external" / "blackjack_hands" / "card_distribution.json"


SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}



def _normalise_prefix(value: str) -> str:
    filtered = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in value)
    filtered = filtered.strip('_').lower()
    return filtered or 'ext'



def _gather_pairs_from_directory(
    image_dir: Path,
    label_dir: Optional[Path] = None,
) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    if not image_dir.exists():
        return pairs

    if label_dir is not None and not label_dir.exists():
        print(f"Warning: label directory missing for {image_dir}")
        return pairs

    missing = 0
    iterable = sorted(image_dir.rglob('*')) if image_dir.is_dir() else []
    for entry in iterable:
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        if label_dir is not None:
            try:
                relative = entry.relative_to(image_dir)
            except ValueError:
                # entry is not within image_dir; skip
                continue
            label_path = (label_dir / relative).with_suffix('.txt')
        else:
            label_path = entry.with_suffix('.txt')
        if not label_path.exists():
            missing += 1
            continue
        pairs.append((entry, label_path))

    if missing:
        print(f"Warning: skipped {missing} images without labels in {image_dir}")
    return pairs



def _harvest_yolo_dataset(root: Path) -> Dict[str, List[Tuple[Path, Path]]]:
    splits: Dict[str, List[Tuple[Path, Path]]] = {"train": [], "val": [], "unsplit": []}
    if not root.exists():
        print(f"Warning: extra dataset path not found: {root}")
        return splits

    images_root = root / 'images'
    labels_root = root / 'labels'
    if images_root.is_dir() and labels_root.is_dir():
        train_pairs = _gather_pairs_from_directory(images_root / 'train', labels_root / 'train')
        val_pairs: List[Tuple[Path, Path]] = []
        for candidate in ('val', 'valid', 'validation'):
            val_pairs.extend(_gather_pairs_from_directory(images_root / candidate, labels_root / candidate))

        if val_pairs:
            splits['val'].extend(val_pairs)
            splits['train'].extend(train_pairs)
        elif train_pairs:
            splits['unsplit'].extend(train_pairs)
        else:
            splits['unsplit'].extend(_gather_pairs_from_directory(images_root, labels_root))
        return splits

    train_dir = root / 'train'
    train_pairs = _gather_pairs_from_directory(train_dir) if train_dir.is_dir() else []

    val_pairs: List[Tuple[Path, Path]] = []
    for candidate in ('val', 'valid', 'validation'):
        candidate_dir = root / candidate
        if candidate_dir.is_dir():
            val_pairs.extend(_gather_pairs_from_directory(candidate_dir))

    if val_pairs:
        splits['val'].extend(val_pairs)
        if train_pairs:
            splits['train'].extend(train_pairs)
    elif train_pairs:
        splits['unsplit'].extend(train_pairs)
    else:
        splits['unsplit'].extend(_gather_pairs_from_directory(root))

    return splits



def _copy_pairs_to_destination(
    pairs: Sequence[Tuple[Path, Path]],
    destination: Path,
    prefix: str,
) -> int:
    if not pairs:
        return 0

    destination.mkdir(parents=True, exist_ok=True)
    copied = 0

    for image_path, label_path in pairs:
        unique_base = f"{prefix}_{uuid.uuid4().hex}"
        dest_image = destination / f"{unique_base}{image_path.suffix.lower()}"
        dest_label = destination / f"{unique_base}.txt"
        shutil.copy2(image_path, dest_image)
        shutil.copy2(label_path, dest_label)
        copied += 1

    return copied



def merge_external_datasets(
    train_dir: Path,
    val_dir: Path,
    dataset_roots: Sequence[Path],
    train_dirs: Sequence[Path],
    val_dirs: Sequence[Path],
    seed: int,
    auto_val_split: float,
) -> Tuple[int, int, List[str]]:
    if dataset_roots and not 0.0 < auto_val_split < 1.0:
        raise ValueError('extra_dataset_val_split must be between 0 and 1 (exclusive)')

    total_train = 0
    total_val = 0
    summaries: List[str] = []
    rng = random.Random(seed)

    for root in dataset_roots:
        splits = _harvest_yolo_dataset(root)
        train_pairs = list(splits['train'])
        val_pairs = list(splits['val'])
        unsplit_pairs = list(splits['unsplit'])

        if unsplit_pairs:
            if len(unsplit_pairs) == 1:
                train_pairs.extend(unsplit_pairs)
            else:
                unsplit_pairs_copy = unsplit_pairs.copy()
                rng.shuffle(unsplit_pairs_copy)
                val_fraction = auto_val_split
                if not 0.0 < val_fraction < 1.0:
                    raise ValueError('extra_dataset_val_split must be between 0 and 1 (exclusive)')
                val_count = int(round(len(unsplit_pairs_copy) * val_fraction))
                val_count = max(1, min(val_count, len(unsplit_pairs_copy) - 1))
                train_count = len(unsplit_pairs_copy) - val_count
                train_pairs.extend(unsplit_pairs_copy[:train_count])
                val_pairs.extend(unsplit_pairs_copy[train_count:])

        prefix = _normalise_prefix(root.name or root.parent.name)
        added_train = _copy_pairs_to_destination(train_pairs, train_dir, prefix)
        added_val = _copy_pairs_to_destination(val_pairs, val_dir, prefix)
        total_train += added_train
        total_val += added_val
        if added_train or added_val:
            summaries.append(f"{root}: train +{added_train}, val +{added_val}")

    for extra_dir in train_dirs:
        pairs = _gather_pairs_from_directory(extra_dir)
        prefix = _normalise_prefix(extra_dir.name)
        added = _copy_pairs_to_destination(pairs, train_dir, prefix)
        total_train += added
        if added:
            summaries.append(f"{extra_dir}: train +{added}")

    for extra_dir in val_dirs:
        pairs = _gather_pairs_from_directory(extra_dir)
        prefix = _normalise_prefix(extra_dir.name)
        added = _copy_pairs_to_destination(pairs, val_dir, prefix)
        total_val += added
        if added:
            summaries.append(f"{extra_dir}: val +{added}")

    return total_train, total_val, summaries


class CardSampler:
    def __init__(self, card_names: Sequence[str], weights: Optional[Dict[str, float]] = None) -> None:
        self._cards: List[str] = list(card_names)
        self._uniform = True
        self._cum_weights: List[float] = []
        self._total_weight = 0.0

        weights = weights or {}
        if not weights:
            return

        lookup = _build_weight_lookup(weights)
        cumulative: List[float] = []
        filtered_cards: List[str] = []
        total = 0.0
        for name in card_names:
            weight = _weight_for_card(name, lookup)
            if weight <= 0.0:
                continue
            total += weight
            cumulative.append(total)
            filtered_cards.append(name)

        if not filtered_cards or total <= 0.0:
            return

        self._cards = filtered_cards
        self._cum_weights = cumulative
        self._total_weight = total
        self._uniform = False

    def sample(self, rng: random.Random) -> str:
        if not self._cards:
            raise RuntimeError('CardSampler has no cards to sample from')
        if self._uniform or self._total_weight <= 0.0:
            return rng.choice(self._cards)
        value = rng.random() * self._total_weight
        idx = bisect.bisect_left(self._cum_weights, value)
        if idx >= len(self._cards):
            idx = len(self._cards) - 1
        return self._cards[idx]


def _build_weight_lookup(weights: Dict[str, float]) -> Dict[str, float]:
    lookup: Dict[str, float] = {}
    for key, raw_value in weights.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value <= 0.0:
            continue
        key_norm = str(key).lower()
        lookup[key_norm] = value
        if key_norm.isdigit():
            lookup[str(int(key_norm))] = value
    if '10' in lookup:
        ten_weight = lookup['10']
        for face in ('jack', 'queen', 'king'):
            lookup.setdefault(face, ten_weight)
    ace_weight = lookup.get('ace') or lookup.get('11')
    if ace_weight:
        lookup['ace'] = ace_weight
        lookup['11'] = ace_weight
    if 'joker' in lookup:
        joker_weight = lookup['joker']
        lookup.setdefault('black', joker_weight)
        lookup.setdefault('red', joker_weight)
        lookup.setdefault('black_joker', joker_weight)
        lookup.setdefault('red_joker', joker_weight)
    return lookup


def _weight_for_card(card_name: str, lookup: Dict[str, float]) -> float:
    lower = card_name.lower()
    rank = lower.split('_')[0]
    candidates = [lower, rank]
    if 'joker' in lower:
        candidates.extend(['joker', 'black_joker', 'red_joker'])
    value = DEFAULT_VALUE_LOOKUP.get(rank)
    if value is not None:
        candidates.append(str(value))
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return 1.0


def load_card_distribution(path: Optional[Path]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.exists():
        print(f'Card distribution file not found: {resolved}')
        return None
    try:
        data = json.loads(resolved.read_text())
    except json.JSONDecodeError as exc:
        print(f'Failed to parse card distribution JSON {resolved}: {exc}')
        return None

    combined: Dict[str, float] = {}
    for key in ('player_card_counts', 'dealer_card_counts'):
        counts = data.get(key)
        if isinstance(counts, dict):
            for rank, raw_count in counts.items():
                try:
                    combined[rank] = combined.get(rank, 0.0) + float(raw_count)
                except (TypeError, ValueError):
                    continue

    total = sum(combined.values())
    if total <= 0.0:
        print(f'Card distribution file {resolved} does not contain usable counts.')
        return None

    weights = {rank: count / total for rank, count in combined.items()}
    print(f'Loaded card distribution weights from {resolved}')
    return weights



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic dataset generation and YOLOv8 training pipeline for Blackjack card detection"
    )
    parser.add_argument("--card-dir", type=Path, default=DEFAULT_CARD_DIR, help="Path to transparent card PNGs")
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BACKGROUND_DIR, help="Path holding table background images")
    parser.add_argument("--generated-dir", type=Path, default=GENERATED_DIR, help="Directory where synthetic samples are written")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR, help="Output directory for YOLO training images")
    parser.add_argument("--val-dir", type=Path, default=VAL_DIR, help="Output directory for YOLO validation images")
    parser.add_argument("--extra-dataset", action="append", type=Path, default=[], help="Additional YOLO dataset roots to merge (expects images/ and labels/ subdirectories or train/val folders).")
    parser.add_argument("--extra-train-dir", action="append", type=Path, default=[], help="Directories containing image/label pairs to append to the train split.")
    parser.add_argument("--extra-val-dir", action="append", type=Path, default=[], help="Directories containing image/label pairs to append to the validation split.")
    parser.add_argument("--extra-dataset-val-split", type=float, default=0.2, help="Validation fraction used when extra datasets do not provide an explicit val split.")
    parser.add_argument("--use-kaggle-cards", dest="use_kaggle_cards", action="store_true", help="Automatically merge the converted Kaggle cards dataset when available.")
    parser.add_argument("--no-kaggle-cards", dest="use_kaggle_cards", action="store_false", help="Disable automatic inclusion of the Kaggle cards dataset.")
    parser.add_argument("--card-distribution-json", type=Path, default=DEFAULT_CARD_DISTRIBUTION, help="Optional JSON file with card frequency statistics (see tools/prepare_blackjack_hands.py).")
    parser.add_argument("--disable-card-distribution", action="store_true", help="Ignore any card distribution file even if provided.")
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
    parser.set_defaults(regenerate=True, amp=True, use_kaggle_cards=True)
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
    card_sampler: CardSampler,
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
        card_name = card_sampler.sample(rng)
        card = card_images.get(card_name)
        if card is None:
            continue
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
    card_images: Dict[str, np.ndarray],
    backgrounds: List[np.ndarray],
    card_lookup: Dict[str, int],
    card_sampler: CardSampler,
) -> None:
    rng = random.Random(args.seed)
    args.generated_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.num_images):
        image, labels = synthesise_scene(
            args.img_width,
            args.img_height,
            card_sampler,
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
    distribution_weights = None
    if not args.disable_card_distribution:
        distribution_weights = load_card_distribution(args.card_distribution_json)
    if distribution_weights is None:
        print("Card sampling uses uniform distribution (no weighting applied).")
    card_sampler = CardSampler(card_names, distribution_weights)

    if args.regenerate:
        print("Regenerating dataset and resetting splits...")
        clean_generation_directories(args.generated_dir, args.train_dir, args.val_dir)
        remove_yolo_caches(args.generated_dir.parent)
        generate_dataset(args, card_images, backgrounds, card_lookup, card_sampler)
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

    dataset_roots = [Path(p) for p in args.extra_dataset]
    if args.use_kaggle_cards:
        if KAGGLE_CARDS_YOLO_DIR.exists():
            dataset_roots.append(KAGGLE_CARDS_YOLO_DIR)
        else:
            print(f'Skipping Kaggle cards dataset: {KAGGLE_CARDS_YOLO_DIR} not found.')

    dedup_roots: List[Path] = []
    seen_roots = set()
    for root in dataset_roots:
        resolved = Path(root).resolve()
        if resolved in seen_roots:
            continue
        seen_roots.add(resolved)
        dedup_roots.append(Path(root))

    extra_train, extra_val, extra_summaries = merge_external_datasets(
        args.train_dir,
        args.val_dir,
        dedup_roots,
        args.extra_train_dir,
        args.extra_val_dir,
        args.seed,
        args.extra_dataset_val_split,
    )
    if extra_train or extra_val:
        print("Merged external datasets:")
        for line in extra_summaries:
            print(f"  - {line}")
        print(f"Total extra samples -> train: +{extra_train}, val: +{extra_val}")

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

    trainer = getattr(model, "trainer", None)
    best_weights = getattr(results, "best", None)
    if best_weights is None and trainer is not None:
        best_weights = (
            getattr(trainer, "best", None)
            or getattr(trainer, "best_path", None)
            or getattr(trainer, "best_model_path", None)
        )

    print("Training complete.")
    if best_weights:
        print("Best weights stored at:")
        print(best_weights)
    else:
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir:
            print("Best weights path unavailable from Ultralytics result; check training directory:")
            print(save_dir)
        else:
            print("Best weights path unavailable from Ultralytics result.")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
