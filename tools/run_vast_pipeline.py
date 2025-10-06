"""Orchestrateur "un clic" pour lancer les entraînements Vast.ai vision + advisor."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from typing import List

from tools import vast_train, vast_train_detector
from tools.vast_train import DEFAULT_BASE_URL


def split_cli(value: str | None) -> List[str]:
    if not value:
        return []
    return shlex.split(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lancer automatiquement les deux entraînements Vast.ai.")
    parser.add_argument("--api-key", default=os.getenv("VAST_API_KEY"), help="Clé API Vast.ai")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL de base de l'API Vast.ai")
    parser.add_argument("--skip-vision", action="store_true", help="Ignorer l'entraînement vision")
    parser.add_argument("--skip-advisor", action="store_true", help="Ignorer l'entraînement advisor")
    parser.add_argument(
        "--vision-options",
        help="Options additionnelles passées à tools/vast_train_detector.py (ex: '--interruptible --max-price 1.5')",
    )
    parser.add_argument(
        "--vision-train-args",
        help="Arguments passés à model/train_model.py (ex: '--num-images 4000 --epochs 150')",
    )
    parser.add_argument(
        "--advisor-options",
        help="Options additionnelles passées à tools/vast_train.py (ex: '--interruptible --num-gpus 1')",
    )
    parser.add_argument(
        "--advisor-train-args",
        help="Arguments passés à train_advanced_advisor.py (ex: '--episodes 800000 --device cuda')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les commandes générées sans créer d'instance",
    )
    return parser.parse_args()


def run_step(label: str, script_name: str, func, argv: List[str], dry_run: bool) -> None:
    print(f"\n=== Étape {label} ===")
    print("Commande générée:", "python", script_name, " ".join(shlex.quote(part) for part in argv))
    if dry_run:
        return
    rc = func(argv)
    if rc not in (None, 0):
        raise SystemExit(f"Étape {label} échouée (code {rc})")


def main() -> int:
    args = parse_args()

    if not args.api_key:
        raise SystemExit("Aucune clé API Vast.ai fournie (argument --api-key ou variable VAST_API_KEY).")

    shared = ["--api-key", args.api_key]
    if args.base_url != DEFAULT_BASE_URL:
        shared += ["--base-url", args.base_url]

    if not args.skip_vision:
        vision_argv = shared + split_cli(args.vision_options)
        train_args = split_cli(args.vision_train_args)
        if train_args:
            vision_argv.append("--")
            vision_argv.extend(train_args)
        run_step("Vision", "tools/vast_train_detector.py", vast_train_detector.main, vision_argv, args.dry_run)

    if not args.skip_advisor:
        advisor_argv = shared + split_cli(args.advisor_options)
        train_args = split_cli(args.advisor_train_args)
        if train_args:
            advisor_argv.append("--")
            advisor_argv.extend(train_args)
        run_step("Advisor", "tools/vast_train.py", vast_train.main, advisor_argv, args.dry_run)

    if args.skip_vision and args.skip_advisor:
        print("Aucune étape à exécuter (vision/advisor toutes deux ignorées).")
    else:
        print("Pipeline Vast.ai déclenché. Surveillez vos instances depuis la console Vast.ai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
