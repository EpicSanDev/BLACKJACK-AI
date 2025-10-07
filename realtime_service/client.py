"""Command-line client for analysing images and submitting data to the server."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests

from blackjack import DEFAULT_GAME_RULES, describe_hand, evaluate_hand, get_expert_advice, normalise_hand
from utils.detection import BlackjackDetector

DEFAULT_MODEL_PATH = Path("runs/detect/blackjack_detector2/weights/best.pt")
DEFAULT_SERVER = "http://localhost:8000/api/v1"


def _encode_image(path: Path) -> str:
    with path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def _print_detection_summary(player_cards, dealer_card, advice, total, is_soft) -> None:
    player_desc = describe_hand(player_cards)
    dealer_desc = describe_hand([dealer_card])
    softness = "soft" if is_soft else "hard"
    print("Player:", player_desc, f"(total {total}, {softness})")
    print("Dealer:", dealer_desc)
    print("Advisor action:", advice)


def _submit_to_server(
    server_url: str,
    payload: Dict[str, Any],
    *,
    timeout: float = 10.0,
) -> None:
    url = server_url.rstrip("/") + "/samples"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    print("Sample stored with id:", data.get("sample_id"))


def _request_status(server_url: str) -> None:
    url = server_url.rstrip("/") + "/status"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


def _trigger_training(server_url: str) -> None:
    url = server_url.rstrip("/") + "/train"
    response = requests.post(url, timeout=5)
    if response.status_code >= 400:
        print(response.json())
        response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


def _prepare_payload(
    client_id: str,
    player_cards,
    dealer_card,
    detections,
    advice: str,
    *,
    player_action: Optional[str],
    outcome: Optional[str],
    notes: Optional[str],
    include_image: bool,
    image_path: Path,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "client_id": client_id,
        "player_cards": [card.to_strategy_card() for card in player_cards],
        "dealer_card": dealer_card.to_strategy_card() if dealer_card else None,
        "advisor_action": advice,
        "player_action": player_action or advice,
        "round_outcome": outcome,
        "notes": notes,
        "detections": [card.to_payload() for card in detections],
    }

    if include_image:
        payload["image_base64"] = _encode_image(image_path)
        payload["image_format"] = image_path.suffix.lstrip(".") or "png"
    return payload


def _handle_capture(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path or DEFAULT_MODEL_PATH)
    detector = BlackjackDetector(
        model_path,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    detections = detector.detect(image_path)
    player_cards, dealer_card = detector.group_cards(detections)
    if not player_cards or not dealer_card:
        print("Impossible de détecter suffisamment de cartes.")
        return

    player_strategy_cards = [card.to_strategy_card() for card in player_cards]
    dealer_strategy_card = dealer_card.to_strategy_card()
    normalised_player = normalise_hand(player_strategy_cards)
    total, is_soft = evaluate_hand(normalised_player)
    advice = get_expert_advice(player_strategy_cards, dealer_strategy_card, DEFAULT_GAME_RULES)

    _print_detection_summary(normalised_player, dealer_strategy_card, advice, total, is_soft)

    if args.annotate:
        output_dir = Path(args.annotation_dir) if args.annotation_dir else image_path.parent
        annotated_path = detector.annotate_image(image_path, detections, output_dir)
        print("Annotated image saved to:", annotated_path)

    if not args.send:
        return

    payload = _prepare_payload(
        args.client_id,
        player_cards,
        dealer_card,
        detections,
        advice,
        player_action=args.player_action,
        outcome=args.outcome,
        notes=args.notes,
        include_image=args.include_image,
        image_path=image_path,
    )

    try:
        _submit_to_server(args.server_url, payload)
    except requests.RequestException as exc:
        print("Failed to submit sample:", exc)
        raise SystemExit(1) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime Blackjack advisor client")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Analyse an image and optionally submit it")
    capture.add_argument("image", help="Image to analyse")
    capture.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to YOLO weights")
    capture.add_argument("--device", default=None, help="Torch device override")
    capture.add_argument("--conf-threshold", type=float, default=0.35, help="Detection confidence threshold")
    capture.add_argument("--iou-threshold", type=float, default=0.45, help="Detection IOU threshold")
    capture.add_argument("--client-id", default=os.getenv("BLACKJACK_CLIENT_ID", "cli"), help="Client identifier")
    capture.add_argument("--player-action", default=None, help="Actual action taken by the player")
    capture.add_argument("--outcome", default=None, help="Round outcome (win/lose/push)")
    capture.add_argument("--notes", default=None, help="Optional free-form notes")
    capture.add_argument("--send", action="store_true", help="Submit the result to the server")
    capture.add_argument("--include-image", action="store_true", help="Embed the raw image in the submission")
    capture.add_argument("--annotate", action="store_true", help="Save an annotated copy of the image")
    capture.add_argument("--annotation-dir", default=None, help="Where to store annotations (default: alongside the image)")
    capture.add_argument("--server-url", default=DEFAULT_SERVER, help="Realtime server base URL")

    status = subparsers.add_parser("status", help="Display server status")
    status.add_argument("--server-url", default=DEFAULT_SERVER, help="Realtime server base URL")

    train = subparsers.add_parser("train", help="Trigger training on the server")
    train.add_argument("--server-url", default=DEFAULT_SERVER, help="Realtime server base URL")

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "capture":
        _handle_capture(args)
    elif args.command == "status":
        try:
            _request_status(args.server_url)
        except requests.RequestException as exc:
            print("Unable to fetch status:", exc)
            raise SystemExit(1) from exc
    elif args.command == "train":
        try:
            _trigger_training(args.server_url)
        except requests.RequestException as exc:
            print("Unable to trigger training:", exc)
            raise SystemExit(1) from exc
    else:  # pragma: no cover - defensive, should not happen
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
