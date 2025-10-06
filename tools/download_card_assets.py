from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import requests

DEFAULT_BASE_URL = "https://deckofcardsapi.com/static/img"
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "dataset" / "png"
DEFAULT_BACKGROUNDS_DIR = ROOT_DIR / "dataset" / "backgrounds"

RANK_CODE_MAP: Dict[str, str] = {
    "ace": "A",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "0",
    "jack": "J",
    "queen": "Q",
    "king": "K",
}
SUIT_CODE_MAP: Dict[str, str] = {
    "clubs": "C",
    "diamonds": "D",
    "hearts": "H",
    "spades": "S",
}
JOKER_MAP: Dict[str, str] = {
    "black_joker": "X1",
    "red_joker": "X2",
}


def iter_card_pairs() -> List[Tuple[str, str]]:
    suits = ("clubs", "diamonds", "hearts", "spades")
    pairs: List[Tuple[str, str]] = []

    for suit in suits:
        pairs.append((f"10_of_{suit}", RANK_CODE_MAP["10"] + SUIT_CODE_MAP[suit]))

    for rank in ("2", "3", "4", "5", "6", "7", "8", "9"):
        for suit in suits:
            pairs.append((f"{rank}_of_{suit}", RANK_CODE_MAP[rank] + SUIT_CODE_MAP[suit]))

    for suit in suits:
        pairs.append((f"ace_of_{suit}", RANK_CODE_MAP["ace"] + SUIT_CODE_MAP[suit]))

    pairs.append(("black_joker", JOKER_MAP["black_joker"]))

    for suit in suits:
        pairs.append((f"jack_of_{suit}", RANK_CODE_MAP["jack"] + SUIT_CODE_MAP[suit]))

    for suit in suits:
        pairs.append((f"king_of_{suit}", RANK_CODE_MAP["king"] + SUIT_CODE_MAP[suit]))

    for suit in suits:
        pairs.append((f"queen_of_{suit}", RANK_CODE_MAP["queen"] + SUIT_CODE_MAP[suit]))

    pairs.append(("red_joker", JOKER_MAP["red_joker"]))
    return pairs


def download_card_png(
    name: str,
    code: str,
    *,
    target_dir: Path,
    base_url: str,
    force: bool,
    timeout: float,
    retries: int,
) -> bool:
    target_path = target_dir / f"{name}.png"
    if target_path.exists() and not force:
        return False

    url = f"{base_url.rstrip('/')}/{code}.png"
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"Décodage PNG impossible pour {name} ({code}).")
            if image.ndim == 2:
                image = image[..., np.newaxis]
            if image.shape[2] == 3:
                alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
                image = np.dstack([image, alpha])
            elif image.shape[2] != 4:
                raise RuntimeError(f"Format inattendu pour {name} ({code}). Shape={image.shape}")
            target_dir.mkdir(parents=True, exist_ok=True)
            # cv2.imwrite normalise les métadonnées et supprime les profils ICC corrompus.
            if not cv2.imwrite(str(target_path), image):
                raise RuntimeError(f"Écriture PNG impossible pour {target_path}")
            return True
        except (requests.RequestException, RuntimeError) as exc:
            if attempt >= retries:
                raise
            time.sleep(min(2 ** attempt, 10))
    return False


def ensure_backgrounds(directory: Path, force: bool) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    expected = ("felt.png", "wood.png", "granite.png")
    if not force and all((directory / name).exists() for name in expected):
        return 0

    rng = np.random.default_rng(42)

    felt = np.zeros((1080, 1920, 3), dtype=np.uint8)
    felt[:] = (16, 100, 24)
    felt_noise = rng.normal(0, 6.0, size=felt.shape).astype(np.int16)
    felt = np.clip(felt.astype(np.int16) + felt_noise, 0, 255).astype(np.uint8)

    base = np.linspace(70, 120, felt.shape[1], dtype=np.uint8)
    grad = np.tile(base, (felt.shape[0], 1))
    wood = cv2.applyColorMap(grad, cv2.COLORMAP_AUTUMN)
    wood = cv2.GaussianBlur(wood, (0, 0), 3)

    granite = rng.normal(100, 25, size=felt.shape).astype(np.int16)
    granite = np.clip(granite, 0, 255).astype(np.uint8)

    generated = {
        "felt.png": felt,
        "wood.png": wood,
        "granite.png": granite,
    }

    written = 0
    for filename, image in generated.items():
        if cv2.imwrite(str(directory / filename), image):
            written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Télécharge les assets de cartes à jouer nécessaires au pipeline Blackjack AI."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie pour les PNG de cartes (défaut: dataset/png)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL d'où télécharger les assets (défaut: deckofcardsapi.com)",
    )
    parser.add_argument("--force", action="store_true", help="Réécrit les fichiers existants")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout HTTP en secondes")
    parser.add_argument("--retries", type=int, default=4, help="Nombre de tentatives sur les erreurs transitoires")
    parser.add_argument(
        "--skip-backgrounds",
        action="store_true",
        help="Ne pas générer de textures de fond synthétiques",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    downloaded = 0
    try:
        for name, code in iter_card_pairs():
            if download_card_png(
                name,
                code,
                target_dir=args.output,
                base_url=args.base_url,
                force=args.force,
                timeout=args.timeout,
                retries=args.retries,
            ):
                downloaded += 1
    except Exception as exc:  # pragma: no cover - script utilitaire
        print(f"Erreur lors du téléchargement de {name}: {exc}", file=sys.stderr)
        return 1

    print(f"Assets cartes prêts ({downloaded} fichiers nouveaux/rafraîchis).")

    if not args.skip_backgrounds:
        created = ensure_backgrounds(DEFAULT_BACKGROUNDS_DIR, args.force)
        if created:
            print(f"Textures de fond générées ({created} fichiers).")
        else:
            print("Textures de fond déjà présentes, rien à faire.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
