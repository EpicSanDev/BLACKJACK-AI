from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
PY_BIN = sys.executable


class StepError(RuntimeError):
    pass


def split_cli(raw: str | None) -> List[str]:
    return shlex.split(raw) if raw else []


def run_step(label: str, command: List[str], *, dry_run: bool) -> None:
    print(f"\n=== {label} ===")
    print("Command:", " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return
    result = subprocess.run(command, cwd=ROOT_DIR)
    if result.returncode != 0:
        raise StepError(f"Step '{label}' failed with exit code {result.returncode}.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click pipeline for local and Vast AI blackjack detector training")
    parser.add_argument("--local", dest="run_local", action="store_true", help="Run the local training pipeline (default)")
    parser.add_argument("--no-local", dest="run_local", action="store_false", help="Skip the local training pipeline")
    parser.add_argument("--vast", dest="run_vast", action="store_true", help="Run the Vast.ai pipeline as part of the workflow")
    parser.add_argument("--no-vast", dest="run_vast", action="store_false", help="Skip the Vast.ai pipeline (default)")
    parser.add_argument("--skip-kaggle-download", action="store_true", help="Do not download Kaggle datasets")
    parser.add_argument("--skip-kaggle-convert", action="store_true", help="Do not convert the Kaggle cards dataset to YOLO format")
    parser.add_argument("--skip-card-distribution", action="store_true", help="Do not regenerate the blackjack hand distribution JSON")
    parser.add_argument("--force-kaggle", action="store_true", help="Force re-download of Kaggle datasets")
    parser.add_argument("--card-distribution-sample", type=float, default=0.05, help="Sampling ratio used when summarising blackjack hands (0 < r <= 1)")
    parser.add_argument("--train-args", help="Additional arguments forwarded to model/train_model.py")
    parser.add_argument("--vast-args", help="Additional arguments forwarded to tools/run_vast_pipeline.py")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned commands without executing them")
    parser.set_defaults(run_local=True, run_vast=False)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.run_local and not args.run_vast:
        print("Nothing to do: both local and Vast pipelines are disabled.")
        return 0

    if args.card_distribution_sample <= 0.0 or args.card_distribution_sample > 1.0:
        raise SystemExit("--card-distribution-sample must be within (0, 1]")

    try:
        if args.run_local:
            if not args.skip_kaggle_download:
                download_cmd = [
                    PY_BIN,
                    "tools/download_kaggle_datasets.py",
                ]
                if args.force_kaggle:
                    download_cmd.append("--force")
                run_step("Download Kaggle datasets", download_cmd, dry_run=args.dry_run)

            if not args.skip_kaggle_convert:
                convert_cmd = [
                    PY_BIN,
                    "tools/prepare_kaggle_cards.py",
                    "--overwrite",
                ]
                run_step("Convert Kaggle cards dataset", convert_cmd, dry_run=args.dry_run)

            if not args.skip_card_distribution:
                distribution_cmd = [
                    PY_BIN,
                    "tools/prepare_blackjack_hands.py",
                    f"--sample={args.card_distribution_sample}",
                ]
                run_step("Summarise blackjack hands dataset", distribution_cmd, dry_run=args.dry_run)

            train_cmd = [
                PY_BIN,
                "model/train_model.py",
            ] + split_cli(args.train_args)
            run_step("Local detector training", train_cmd, dry_run=args.dry_run)

        if args.run_vast:
            vast_cmd = [
                PY_BIN,
                "tools/run_vast_pipeline.py",
            ] + split_cli(args.vast_args)
            run_step("Vast.ai pipeline", vast_cmd, dry_run=args.dry_run)

    except StepError as exc:
        print(exc)
        return 1

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
