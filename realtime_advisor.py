from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

from blackjack import ACTION_LABELS_FR, DEFAULT_GAME_RULES, describe_hand, evaluate_hand, get_expert_advice, normalise_hand
from blackjack.advanced_advisor import AdvancedAdvisor
from utils import select_best_device

MODEL_PATH = "runs/detect/blackjack_detector2/weights/best.pt"
DEFAULT_PLAYER_ROI = (600, 800, 400, 200)
DEFAULT_DEALER_ROI = (300, 800, 200, 200)


@dataclass(frozen=True)
class Region:
    name: str
    top: int
    left: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


@dataclass
class DetectedCard:
    rank: str
    value: int
    confidence: float
    global_center: Tuple[float, float]
    global_bbox: Tuple[float, float, float, float]
    local_bbox: Tuple[float, float, float, float]

    def to_strategy_card(self) -> Dict[str, object]:
        return {"rank": self.rank, "value": self.value}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time Blackjack advisor powered by YOLOv8")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to the trained YOLO model")
    parser.add_argument("--player-roi", type=int, nargs=4, default=DEFAULT_PLAYER_ROI, metavar=("TOP", "LEFT", "WIDTH", "HEIGHT"), help="Player capture region")
    parser.add_argument("--dealer-roi", type=int, nargs=4, default=DEFAULT_DEALER_ROI, metavar=("TOP", "LEFT", "WIDTH", "HEIGHT"), help="Dealer capture region")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index as reported by mss")
    parser.add_argument("--margin", type=int, default=60, help="Extra padding (pixels) around combined ROI capture")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size")
    parser.add_argument("--interval", type=float, default=0.35, help="Seconds to wait between frames")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (e.g. '0', 'cuda', 'mps')")
    parser.add_argument("--history", type=int, default=3, help="Frames to smooth advice decisions")
    parser.add_argument("--view", action="store_true", help="Display a debug window with detections")
    parser.add_argument("--full-screen", action="store_true", help="Analyse the full monitor instead of cropped ROI")
    parser.add_argument("--no-surrender", dest="surrender", action="store_false", help="Disable surrender suggestions")
    parser.add_argument("--no-double-after-split", dest="double_after_split", action="store_false", help="Disallow double after split")
    parser.add_argument("--dealer-hits-soft-17", dest="dealer_hits_soft_17", action="store_true", help="Enable dealer hit on soft 17 rule")
    parser.add_argument("--advanced-policy", type=str, default=None, help="Optional JSON policy trained via train_advanced_advisor.py")
    parser.set_defaults(
        surrender=DEFAULT_GAME_RULES["surrender_allowed"],
        double_after_split=DEFAULT_GAME_RULES["double_after_split_allowed"],
        dealer_hits_soft_17=DEFAULT_GAME_RULES["dealer_hits_on_soft_17"],
    )
    return parser.parse_args()


def combined_capture(regions: Sequence[Region], monitor: Dict[str, int], margin: int) -> Dict[str, int]:
    if not regions:
        return monitor
    left = min(region.left for region in regions) - margin
    top = min(region.top for region in regions) - margin
    right = max(region.right for region in regions) + margin
    bottom = max(region.bottom for region in regions) + margin

    left = max(monitor["left"], left)
    top = max(monitor["top"], top)
    right = min(monitor["left"] + monitor["width"], right)
    bottom = min(monitor["top"] + monitor["height"], bottom)

    return {"left": left, "top": top, "width": max(1, right - left), "height": max(1, bottom - top)}


DEFAULT_VALUE_LOOKUP = {
    "ace": 11,
    "king": 10,
    "queen": 10,
    "jack": 10,
    "10": 10,
    "9": 9,
    "8": 8,
    "7": 7,
    "6": 6,
    "5": 5,
    "4": 4,
    "3": 3,
    "2": 2,
    "black": 0,
    "red": 0,
}


def extract_cards(
    detections: Sequence,
    class_names: Dict[int, str],
    capture_left: int,
    capture_top: int,
) -> List[DetectedCard]:
    cards: List[DetectedCard] = []
    for result in detections:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = class_names.get(cls_id, "unknown")
            rank = class_name.split("_")[0]
            value = DEFAULT_VALUE_LOOKUP.get(rank, 0)
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            global_x1 = capture_left + x1
            global_y1 = capture_top + y1
            global_x2 = capture_left + x2
            global_y2 = capture_top + y2
            cards.append(
                DetectedCard(
                    rank=rank,
                    value=value,
                    confidence=conf,
                    global_center=((global_x1 + global_x2) / 2, (global_y1 + global_y2) / 2),
                    global_bbox=(global_x1, global_y1, global_x2, global_y2),
                    local_bbox=(x1, y1, x2, y2),
                )
            )
    return cards


def assign_cards(cards: Sequence[DetectedCard], regions: Sequence[Region]) -> Tuple[Dict[str, List[DetectedCard]], List[DetectedCard]]:
    grouped: Dict[str, List[DetectedCard]] = {region.name: [] for region in regions}
    leftovers: List[DetectedCard] = []
    for card in cards:
        assigned = False
        for region in regions:
            if region.contains(*card.global_center):
                grouped[region.name].append(card)
                assigned = True
                break
        if not assigned:
            leftovers.append(card)
    return grouped, leftovers


def choose_dealer_card(dealer_cards: Sequence[DetectedCard], fallback: Sequence[DetectedCard]) -> Optional[DetectedCard]:
    if dealer_cards:
        return min(dealer_cards, key=lambda c: c.global_center[1])
    if fallback:
        return min(fallback, key=lambda c: c.global_center[1])
    return None


class AdviceSmoother:
    def __init__(self, window: int) -> None:
        self.history: deque[str] = deque(maxlen=max(1, window))

    def update(self, advice: str) -> str:
        self.history.append(advice)
        counts = Counter(self.history)
        return counts.most_common(1)[0][0]


def runtime_rules(args: argparse.Namespace) -> Dict[str, bool]:
    return {
        "dealer_hits_on_soft_17": bool(args.dealer_hits_soft_17),
        "double_after_split_allowed": bool(args.double_after_split),
        "surrender_allowed": bool(args.surrender),
    }


def render_overlay(
    frame: np.ndarray,
    cards: Sequence[DetectedCard],
    player_region: Region,
    dealer_region: Region,
    advice_text: Optional[str],
    origin: Tuple[int, int],
) -> np.ndarray:
    annotated = frame.copy()
    origin_x, origin_y = origin
    height, width = annotated.shape[:2]

    for card in cards:
        x1, y1, x2, y2 = map(int, card.local_bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(annotated, f"{card.rank} {card.confidence:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    for region, color in ((player_region, (255, 0, 0)), (dealer_region, (0, 0, 255))):
        x1 = max(0, min(width - 1, region.left - origin_x))
        y1 = max(0, min(height - 1, region.top - origin_y))
        x2 = max(0, min(width - 1, region.right - origin_x))
        y2 = max(0, min(height - 1, region.bottom - origin_y))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

    if advice_text:
        cv2.putText(annotated, advice_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return annotated


def main() -> None:
    args = parse_args()

    regions = [Region("player", *args.player_roi), Region("dealer", *args.dealer_roi)]
    rules = runtime_rules(args)
    smoother = AdviceSmoother(args.history)
    device = select_best_device(args.device)

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

    def shutdown_handler(signum, frame):  # noqa: ARG001 - signal handler signature
        print("\nArrêt de l'assistant.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    print("Assistant Blackjack en temps réel prêt. Appuyez sur Ctrl+C pour arrêter.")

    with mss() as sct:
        monitor = sct.monitors[args.monitor]
        capture_region = monitor if args.full_screen else combined_capture(regions, monitor, args.margin)
        origin = (capture_region["left"], capture_region["top"])

        while True:
            start_time = time.perf_counter()
            screenshot = sct.grab(capture_region)
            frame_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

            results = model.predict(
                source=frame_bgr,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )

            cards = extract_cards(results, model.names, capture_region["left"], capture_region["top"])
            grouped, leftovers = assign_cards(cards, regions)
            player_cards = grouped.get("player", [])
            dealer_cards = grouped.get("dealer", [])
            dealer_card = choose_dealer_card(dealer_cards, leftovers)

            advice_output = "En attente de cartes..."
            overlay_text: Optional[str] = None

            if player_cards and dealer_card:
                strategy_player_cards = [card.to_strategy_card() for card in player_cards]
                strategy_dealer_card = dealer_card.to_strategy_card()
                if advanced_advisor is not None:
                    advice = advanced_advisor.recommend(strategy_player_cards, strategy_dealer_card)
                else:
                    advice = get_expert_advice(strategy_player_cards, strategy_dealer_card, rules)
                smoothed = smoother.update(advice)

                normalised_player = normalise_hand(strategy_player_cards)
                total, is_soft = evaluate_hand(normalised_player)
                player_desc = describe_hand(normalised_player)
                dealer_desc = describe_hand([strategy_dealer_card])
                advice_label = ACTION_LABELS_FR.get(smoothed, smoothed)
                soft_suffix = " (soft)" if is_soft else ""
                advice_output = (
                    f"Joueur: {total:>2}{soft_suffix} ({player_desc}) | "
                    f"Croupier: {strategy_dealer_card['value']:>2} ({dealer_desc}) -> CONSEIL: {advice_label:<10}"
                )
                overlay_text = advice_label
            elif player_cards or dealer_card:
                advice_output = "Détection partielle, en attente de toutes les cartes..."

            print(advice_output.ljust(120), end="\r", flush=True)

            if args.view:
                annotated = render_overlay(frame_bgr, cards, regions[0], regions[1], overlay_text, origin)
                cv2.imshow("Blackjack Advisor", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            elapsed = time.perf_counter() - start_time
            time.sleep(max(0.0, args.interval - elapsed))

    if args.view:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
