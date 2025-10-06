from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import cv2
from flask import Flask, redirect, render_template, request, url_for
from ultralytics import YOLO

from blackjack import ACTION_LABELS_FR, DEFAULT_GAME_RULES, describe_hand, evaluate_hand, get_expert_advice, normalise_hand
from utils import select_best_device

app = Flask(__name__)

MODEL_PATH = Path("runs/detect/blackjack_detector2/weights/best.pt")
UPLOAD_FOLDER = Path("app/static/uploads")
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
DEVICE = select_best_device(os.environ.get("YOLO_DEVICE"))

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

print(f"Loading YOLO model from {MODEL_PATH} on device {DEVICE}...")
model = YOLO(str(MODEL_PATH))
model.to(DEVICE)

CARD_VALUE_MAP: Dict[str, int] = {
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


def _extract_cards(image_path: str) -> List[Dict[str, object]]:
    results = model.predict(
        source=image_path,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )
    cards: List[Dict[str, object]] = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[int(cls_id)]
            rank = class_name.split("_")[0]
            value = CARD_VALUE_MAP.get(rank, 0)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cards.append(
                {
                    "rank": rank,
                    "value": value,
                    "box": (float(x1), float(y1), float(x2), float(y2)),
                    "confidence": float(box.conf[0]),
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                }
            )
    return cards


def _group_cards(cards: List[Dict[str, object]]) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    if not cards:
        return [], {}

    cards_sorted = sorted(cards, key=lambda c: c["center"][1], reverse=True)
    player_cards = cards_sorted[:2]
    dealer_candidates = [c for c in cards_sorted[2:] if c["value"] > 0] or cards_sorted[1:]
    dealer_card = min(dealer_candidates, key=lambda c: c["center"][1]) if dealer_candidates else {}
    return player_cards, dealer_card


def _annotate_image(image_path: str, cards: List[Dict[str, object]]) -> str:
    image = cv2.imread(image_path)
    for card in cards:
        x1, y1, x2, y2 = map(int, card["box"])
        label = f"{card['rank']} ({card['confidence']:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(image, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    filename = "annotated_" + Path(image_path).name
    annotated_path = UPLOAD_FOLDER / filename
    cv2.imwrite(str(annotated_path), image)
    return url_for("static", filename=f"uploads/{filename}")


@app.route("/", methods=["GET", "POST"])
def upload_file():
    advice_text = None
    advice_details = None
    annotated_url = None

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file or file.filename == "":
            return redirect(request.url)

        filepath = UPLOAD_FOLDER / file.filename
        file.save(filepath)

        detections = _extract_cards(str(filepath))
        player_cards, dealer_card = _group_cards(detections)

        if player_cards and dealer_card:
            strategy_player_cards = [
                {"rank": card["rank"], "value": card["value"]}
                for card in player_cards
            ]
            strategy_dealer_card = {"rank": dealer_card["rank"], "value": dealer_card["value"]}
            advice = get_expert_advice(strategy_player_cards, strategy_dealer_card, DEFAULT_GAME_RULES)
            normalised_player = normalise_hand(strategy_player_cards)
            total, is_soft = evaluate_hand(normalised_player)
            player_desc = describe_hand(normalised_player)
            dealer_desc = describe_hand([strategy_dealer_card])

            advice_text = ACTION_LABELS_FR.get(advice, advice)
            advice_details = {
                "player_total": total,
                "player_soft": is_soft,
                "player_cards": player_desc,
                "dealer_card": dealer_desc,
            }
        else:
            advice_text = "Pas assez de cartes détectées."

        annotated_url = _annotate_image(str(filepath), detections)

    return render_template(
        "index.html",
        advice=advice_text,
        details=advice_details,
        image_url=annotated_url,
    )


if __name__ == "__main__":
    app.run(debug=True)
