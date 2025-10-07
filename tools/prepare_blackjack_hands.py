from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, TextIO, List

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT_DIR / "dataset" / "external" / "blackjack_hands_raw" / "blackjack_hands.csv"
DEFAULT_OUTPUT = ROOT_DIR / "dataset" / "external" / "blackjack_hands" / "card_distribution.json"

CARD_PATTERN = re.compile(r"-?\d+")


class ColumnDetectionError(RuntimeError):
    pass


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise the Kaggle blackjack hands dataset into card frequency statistics.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to blackjack_hands.csv or its .zip archive")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination JSON file for aggregated statistics")
    parser.add_argument("--player-column", help="Optional explicit column name for the player's initial cards")
    parser.add_argument("--dealer-column", help="Optional explicit column name for the dealer's up card")
    parser.add_argument("--limit", type=int, help="Maximum number of rows to process")
    parser.add_argument("--sample", type=float, default=1.0, help="Random sampling ratio in (0,1]")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when subsampling the dataset")
    return parser.parse_args(argv)


def open_csv(path: Path) -> TextIO:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        candidates = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not candidates:
            raise RuntimeError(f"No CSV file found inside {path}")
        stream = archive.open(candidates[0], "r")
        wrapper = io.TextIOWrapper(stream, encoding="utf-8")
        setattr(wrapper, "_archive", archive)
        return wrapper
    return path.open("r", encoding="utf-8")


def detect_column(fieldnames: Iterable[str], keywords: Iterable[str]) -> Optional[str]:
    lowered = {name.lower(): name for name in fieldnames if name}
    for lower, original in lowered.items():
        if all(keyword in lower for keyword in keywords):
            return original
    return None


def card_value_to_rank(value: int) -> Optional[str]:
    if value in {1, 11}:
        return "ace"
    if value == 10:
        return "10"
    if 2 <= value <= 9:
        return str(value)
    return None


def parse_card_values(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(match.group()) for match in CARD_PATTERN.finditer(raw)]


def summarise(args: argparse.Namespace) -> None:
    if not 0.0 < args.sample <= 1.0:
        raise ValueError('--sample must be within (0, 1]')
    rng = random.Random(args.seed)
    rows_seen = 0
    rows_used = 0
    player_counts: Counter[str] = Counter()
    dealer_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()

    with open_csv(args.input) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("CSV file has no header row")

        player_col = args.player_column or detect_column(reader.fieldnames, ("player", "initial", "card"))
        dealer_col = args.dealer_column or detect_column(reader.fieldnames, ("dealer", "up"))

        if not player_col or not dealer_col:
            raise ColumnDetectionError(
                "Unable to locate required columns. Use --player-column/--dealer-column to specify them explicitly."
            )

        for row in reader:
            rows_seen += 1
            if args.limit and rows_used >= args.limit:
                break
            if args.sample < 1.0 and rng.random() > args.sample:
                continue

            player_values = parse_card_values(row.get(player_col, ""))
            dealer_values = parse_card_values(row.get(dealer_col, ""))
            if not player_values or not dealer_values:
                continue

            dealer_rank = card_value_to_rank(dealer_values[0])
            if dealer_rank:
                dealer_counts[dealer_rank] += 1

            ranks = [card_value_to_rank(val) for val in player_values if card_value_to_rank(val)]
            if not ranks:
                continue
            rows_used += 1
            for rank in ranks:
                player_counts[rank] += 1
            key = "|".join(sorted(ranks[:2]))
            if key:
                pair_counts[key] += 1

    if not rows_used:
        raise RuntimeError("No usable rows found in the dataset")

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "rows_seen": rows_seen,
        "rows_used": rows_used,
        "player_card_counts": dict(player_counts),
        "dealer_card_counts": dict(dealer_counts),
        "player_pair_counts": dict(pair_counts.most_common()),
        "source": str(args.input),
        "sample_ratio": args.sample,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved card distribution summary to {args.output} (rows used: {rows_used})")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summarise(args)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
