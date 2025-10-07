from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT_DIR / "dataset" / "external" / "kaggle_cards_raw"
DEFAULT_OUTPUT = ROOT_DIR / "dataset" / "external" / "kaggle_cards_yolo"
DATASET_CONFIG = ROOT_DIR / "dataset.yaml"

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "test": "test",
    "valid": "val",
    "validation": "val",
    "val": "val",
}

TEXT_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


class CardMappingError(RuntimeError):
    pass


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Kaggle cards classification dataset into YOLO detection format.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the raw Kaggle cards dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory for YOLO-formatted assets")
    parser.add_argument("--config", type=Path, default=DATASET_CONFIG, help="YOLO dataset YAML used for class names")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory contents")
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optional limit for the number of images per split (useful for quick smoke tests)",
    )
    return parser.parse_args(argv)


def load_class_mapping(config_path: Path) -> Dict[str, int]:
    data = yaml_safe_load(config_path)
    names = data.get("names")
    if not isinstance(names, list):
        raise RuntimeError(f"Unable to read class names from {config_path}")
    return {str(name): idx for idx, name in enumerate(names)}


def yaml_safe_load(path: Path) -> Dict[str, object]:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def normalise_class_name(raw: str) -> str:
    tokenised = []
    for part in raw.replace("/", " ").replace("-", " ").split():
        lower = part.lower()
        tokenised.append(TEXT_TO_DIGIT.get(lower, lower))
    slug = "_".join(tokenised)
    slug = slug.replace("__", "_")
    return slug


def map_to_known_class(raw: str, class_ids: Dict[str, int]) -> Tuple[str, int]:
    slug = normalise_class_name(raw)
    if slug in class_ids:
        return slug, class_ids[slug]

    if slug.endswith("_of_club"):
        slug = slug.replace("_of_club", "_of_clubs")
    elif slug.endswith("_of_diamond"):
        slug = slug.replace("_of_diamond", "_of_diamonds")
    elif slug.endswith("_of_heart"):
        slug = slug.replace("_of_heart", "_of_hearts")
    elif slug.endswith("_of_spade"):
        slug = slug.replace("_of_spade", "_of_spades")

    if slug in class_ids:
        return slug, class_ids[slug]

    if slug == "joker":
        if "black_joker" in class_ids:
            return "black_joker", class_ids["black_joker"]
        if "red_joker" in class_ids:
            return "red_joker", class_ids["red_joker"]

    raise CardMappingError(f"Unknown card class '{raw}' (normalised to '{slug}')")


def prepare_output_dirs(output_root: Path, overwrite: bool) -> Dict[str, Tuple[Path, Path]]:
    if output_root.exists() and overwrite:
        import shutil

        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in {"train", "val", "test"}:
        images_dir = output_root / "images" / split
        labels_dir = output_root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        splits[split] = (images_dir, labels_dir)
    return splits


def detect_card_bbox(image: np.ndarray) -> Tuple[int, int, int, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, image.shape[1], image.shape[0]

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    if area < 0.05 * (image.shape[0] * image.shape[1]):
        return 0, 0, image.shape[1], image.shape[0]
    return x, y, w, h


def write_label(label_path: Path, class_id: int, bbox: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> None:
    x, y, w, h = bbox
    width, height = image_size
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    bw = w / width
    bh = h / height
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 1e-6), 1.0)
    bh = min(max(bh, 1e-6), 1.0)
    label_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")


def iter_split_directories(input_root: Path) -> List[Tuple[str, Path]]:
    splits: List[Tuple[str, Path]] = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        split = SPLIT_ALIASES.get(child.name.lower())
        if not split:
            continue
        splits.append((split, child))
    return splits


def process_dataset(args: argparse.Namespace) -> None:
    class_ids = load_class_mapping(args.config)
    split_dirs = iter_split_directories(args.input)
    if not split_dirs:
        raise SystemExit(f"No split directories found inside {args.input}")

    splits = prepare_output_dirs(args.output, overwrite=args.overwrite)
    stats = {"train": 0, "val": 0, "test": 0}

    for split, split_dir in split_dirs:
        images_dir, labels_dir = splits[split]
        processed = 0
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                class_name, class_id = map_to_known_class(class_dir.name, class_ids)
            except CardMappingError as exc:
                raise SystemExit(str(exc))

            for image_path in sorted(class_dir.glob("*")):
                if not image_path.is_file():
                    continue
                ext = image_path.suffix.lower()
                if ext not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue

                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                x, y, w, h = detect_card_bbox(image)
                base = f"{class_name}_{uuid.uuid4().hex}"
                dest_image = images_dir / f"{base}{ext}"
                dest_label = labels_dir / f"{base}.txt"
                cv2.imwrite(str(dest_image), image)
                write_label(dest_label, class_id, (x, y, w, h), (image.shape[1], image.shape[0]))
                processed += 1
                stats[split] += 1
                if args.max_images and processed >= args.max_images:
                    break
            if args.max_images and processed >= args.max_images:
                break
        print(f"Split {split}: converted {processed} images from {split_dir}")

    summary_path = args.output / "conversion_summary.json"
    summary_path.write_text(json.dumps({"counts": stats}, indent=2), encoding="utf-8")
    print(f"Conversion summary written to {summary_path}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"Input directory not found: {args.input}")
    process_dataset(args)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
