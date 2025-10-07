from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

DATASETS = {
    "cards": {
        "ref": "gpiosenka/cards-image-datasetclassification",
        "default_dir": Path("dataset/external/kaggle_cards_raw"),
        "description": "Cards Image Dataset-Classification",
    },
    "hands": {
        "ref": "dennisho/blackjack-hands",
        "default_dir": Path("dataset/external/blackjack_hands_raw"),
        "description": "50 Million Blackjack Hands",
    },
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the Kaggle card datasets used by the Blackjack AI pipeline.",
    )
    parser.add_argument(
        "--cards-dir",
        type=Path,
        default=DATASETS["cards"]["default_dir"],
        help="Destination directory for the cards image dataset.",
    )
    parser.add_argument(
        "--hands-dir",
        type=Path,
        default=DATASETS["hands"]["default_dir"],
        help="Destination directory for the blackjack hands dataset.",
    )
    parser.add_argument("--skip-cards", action="store_true", help="Skip downloading the cards image dataset.")
    parser.add_argument("--skip-hands", action="store_true", help="Skip downloading the blackjack hands dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove any existing destination folders before downloading.",
    )
    parser.add_argument(
        "--kaggle-cmd",
        default=shutil.which("kaggle") or "kaggle",
        help="Path to the Kaggle CLI executable (defaults to searching PATH).",
    )
    return parser.parse_args(argv)


def ensure_cli_available(binary: str) -> Path:
    resolved = shutil.which(binary)
    if not resolved:
        raise SystemExit(
            "The Kaggle CLI was not found. Install it via 'pip install kaggle' and set KAGGLE_USERNAME/KAGGLE_KEY."
        )
    return Path(resolved)


def clean_destination(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_dataset(kaggle_bin: Path, ref: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    command = [
        str(kaggle_bin),
        "datasets",
        "download",
        "-d",
        ref,
        "-p",
        str(dest),
        "--unzip",
    ]
    print(f"Downloading {ref} into {dest} ...")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Kaggle CLI exited with status {result.returncode} while downloading {ref}.")
    print(f"Finished downloading {ref}.")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    kaggle_bin = ensure_cli_available(args.kaggle_cmd)

    tasks = []
    if not args.skip_cards:
        tasks.append((DATASETS["cards"], args.cards_dir))
    if not args.skip_hands:
        tasks.append((DATASETS["hands"], args.hands_dir))

    if not tasks:
        print("Nothing to do. All downloads skipped.")
        return 0

    for meta, target in tasks:
        if args.force:
            clean_destination(target)
        elif any(target.iterdir()) if target.exists() else False:
            print(f"Destination {target} is not empty. Use --force to refresh it. Skipping {meta['description']}.")
            continue

        download_dataset(kaggle_bin, meta["ref"], target)

    print("All requested Kaggle datasets processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
