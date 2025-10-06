from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from blackjack import (
    ACTION_LABELS_FR,
    DEFAULT_GAME_RULES,
    describe_hand,
    evaluate_hand,
    get_expert_advice,
    normalise_hand,
)
from blackjack.advanced_advisor import AdvancedAdvisor
from realtime_advisor import (
    MODEL_PATH,
    DEFAULT_DEALER_ROI,
    DEFAULT_PLAYER_ROI,
    AdviceSmoother,
    Region,
    assign_cards,
    choose_dealer_card,
    extract_cards,
    render_overlay,
    runtime_rules,
)
from utils import select_best_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blackjack advisor compatible with Google Colab (video or frame sources).",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a video file (e.g. MP4/MOV) or directory of image frames.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to the trained YOLO model (default: %(default)s).",
    )
    parser.add_argument(
        "--player-roi",
        type=int,
        nargs=4,
        metavar=("TOP", "LEFT", "WIDTH", "HEIGHT"),
        default=None,
        help="Player capture region within the source frames.",
    )
    parser.add_argument(
        "--dealer-roi",
        type=int,
        nargs=4,
        metavar=("TOP", "LEFT", "WIDTH", "HEIGHT"),
        default=None,
        help="Dealer capture region within the source frames.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Optional path to save annotated output video (MP4 recommended).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback FPS when --source is a directory of frames (default: %(default)s).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display annotated frames inline using cv2_imshow (requires Google Colab).",
    )
    parser.add_argument(
        "--display-interval",
        type=int,
        default=30,
        help="Display every Nth frame when --display is set (default: %(default)s).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on the number of frames to process.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: %(default)s processes all frames).",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=3,
        help="Frames to smooth advice decisions.",
    )
    parser.add_argument(
        "--advanced-policy",
        type=str,
        default=None,
        help="Optional JSON policy trained via train_advanced_advisor.py.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (e.g. 'cuda', '0', 'cpu').",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="YOLO IoU threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--no-surrender",
        dest="surrender",
        action="store_false",
        help="Disable surrender suggestions.",
    )
    parser.add_argument(
        "--no-double-after-split",
        dest="double_after_split",
        action="store_false",
        help="Disallow double after split.",
    )
    parser.add_argument(
        "--dealer-hits-soft-17",
        dest="dealer_hits_soft_17",
        action="store_true",
        help="Enable dealer hit on soft 17 rule.",
    )
    parser.set_defaults(
        surrender=DEFAULT_GAME_RULES["surrender_allowed"],
        double_after_split=DEFAULT_GAME_RULES["double_after_split_allowed"],
        dealer_hits_soft_17=DEFAULT_GAME_RULES["dealer_hits_on_soft_17"],
    )
    return parser.parse_args()


def iter_frames_from_directory(directory: Path) -> Iterator[np.ndarray]:
    image_paths = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}),
    )
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        yield frame


def iter_frames_from_video(video_path: Path) -> Tuple[Iterator[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0

    def generator() -> Iterator[np.ndarray]:
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                yield frame
        finally:
            capture.release()

    return generator(), fps


def ensure_display_callable(enabled: bool, interval: int):
    if not enabled:
        return None

    try:
        from google.colab.patches import cv2_imshow  # type: ignore[import-not-found]
    except ImportError:
        print("L'affichage inline (--display) n'est pas disponible hors de Google Colab.")
        return None

    interval = max(1, interval)

    def _show(frame: np.ndarray, index: int) -> None:
        if index % interval == 0:
            cv2_imshow(frame)

    return _show


def prepare_regions(args: argparse.Namespace) -> Tuple[Region, Region]:
    player = Region("player", *(args.player_roi or DEFAULT_PLAYER_ROI))
    dealer = Region("dealer", *(args.dealer_roi or DEFAULT_DEALER_ROI))
    return player, dealer


def process_frames(
    frames: Iterable[np.ndarray],
    frame_rate: float,
    player_region: Region,
    dealer_region: Region,
    args: argparse.Namespace,
) -> None:
    device = select_best_device(args.device)
    smoother = AdviceSmoother(args.history)
    rules = runtime_rules(args)

    advanced_advisor: Optional[AdvancedAdvisor] = None
    if args.advanced_policy:
        try:
            advanced_advisor = AdvancedAdvisor.from_file(args.advanced_policy, rules)
            print(f"Politique avancée chargée depuis {args.advanced_policy}.")
        except Exception as exc:  # noqa: BLE001 - surface failure to the user
            print(f"Impossible de charger la politique avancée: {exc}")

    print("Chargement du modèle YOLO...")
    model = YOLO(args.model)
    model.to(device)

    stride = max(1, args.stride)
    display_fn = ensure_display_callable(args.display, args.display_interval)
    writer: Optional[cv2.VideoWriter] = None
    target_fps = frame_rate if frame_rate > 0 else float(args.fps)

    def maybe_init_writer(frame_shape: Tuple[int, int], path: str) -> cv2.VideoWriter:
        nonlocal writer
        if writer is not None:
            return writer
        height, width = frame_shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, target_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Impossible d'initialiser l'écriture vidéo vers {path}")
        return writer

    frame_counter = 0
    processed_counter = 0
    for frame in frames:
        frame_counter += 1
        if args.max_frames is not None and processed_counter >= args.max_frames:
            break
        if (frame_counter - 1) % stride != 0:
            continue

        processed_counter += 1
        frame_bgr = frame.copy()

        results = model.predict(
            source=frame_bgr,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )

        cards = extract_cards(results, model.names, 0, 0)
        grouped, leftovers = assign_cards(cards, (player_region, dealer_region))
        player_cards = grouped.get("player", [])
        dealer_cards = grouped.get("dealer", [])
        dealer_card = choose_dealer_card(dealer_cards, leftovers)

        advice_output = "En attente de cartes..."
        overlay_text: Optional[str] = None

        if player_cards and dealer_card:
            strategy_player = [card.to_strategy_card() for card in player_cards]
            strategy_dealer = dealer_card.to_strategy_card()
            if advanced_advisor is not None:
                advice = advanced_advisor.recommend(strategy_player, strategy_dealer)
            else:
                advice = get_expert_advice(strategy_player, strategy_dealer, rules)
            smoothed = smoother.update(advice)

            normalised_player = normalise_hand(strategy_player)
            total, is_soft = evaluate_hand(normalised_player)
            player_desc = describe_hand(normalised_player)
            dealer_desc = describe_hand([strategy_dealer])
            advice_label = ACTION_LABELS_FR.get(smoothed, smoothed)
            soft_suffix = " (soft)" if is_soft else ""
            advice_output = (
                f"Frame {frame_counter:05d} -> Joueur: {total:>2}{soft_suffix} ({player_desc}) | "
                f"Croupier: {strategy_dealer['value']:>2} ({dealer_desc}) -> CONSEIL: {advice_label:<10}"
            )
            overlay_text = advice_label
        elif player_cards or dealer_card:
            advice_output = f"Frame {frame_counter:05d} -> Détection partielle, en attente de toutes les cartes..."
        else:
            advice_output = f"Frame {frame_counter:05d} -> Aucune carte détectée."

        print(advice_output)

        annotated = render_overlay(frame_bgr, cards, player_region, dealer_region, overlay_text, origin=(0, 0))

        if args.output_video:
            writer_instance = maybe_init_writer(annotated.shape[:2], args.output_video)
            writer_instance.write(annotated)

        if display_fn is not None:
            display_fn(annotated, frame_counter)

    if writer is not None:
        writer.release()

    print("Traitement terminé.")


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)

    if not source_path.exists():
        print(f"Source introuvable: {source_path}")
        sys.exit(1)

    player_region, dealer_region = prepare_regions(args)

    if source_path.is_dir():
        frames = iter_frames_from_directory(source_path)
        fps = float(args.fps)
    else:
        frames, fps = iter_frames_from_video(source_path)
        if fps <= 0:
            fps = float(args.fps)

    process_frames(frames, fps, player_region, dealer_region, args)


if __name__ == "__main__":
    main()
