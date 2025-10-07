from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for

from blackjack import ACTION_LABELS_FR, DEFAULT_GAME_RULES, describe_hand, evaluate_hand, get_expert_advice, normalise_hand
from utils import select_best_device
from utils.detection import BlackjackDetector

app = Flask(__name__)

MODEL_PATH = Path("runs/detect/blackjack_detector2/weights/best.pt")
UPLOAD_FOLDER = Path("app/static/uploads")
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
DEVICE = select_best_device(os.environ.get("YOLO_DEVICE"))

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

print(f"Loading YOLO model from {MODEL_PATH} on device {DEVICE}...")
detector = BlackjackDetector(
    MODEL_PATH,
    device=DEVICE,
    conf_threshold=CONF_THRESHOLD,
    iou_threshold=IOU_THRESHOLD,
)


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

        detections = detector.detect(filepath)
        player_cards, dealer_card = detector.group_cards(detections)

        if player_cards and dealer_card:
            strategy_player_cards = [card.to_strategy_card() for card in player_cards]
            strategy_dealer_card = dealer_card.to_strategy_card()
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

        annotated_path = detector.annotate_image(filepath, detections, UPLOAD_FOLDER)
        annotated_url = url_for("static", filename=f"uploads/{annotated_path.name}")

    return render_template(
        "index.html",
        advice=advice_text,
        details=advice_details,
        image_url=annotated_url,
    )


if __name__ == "__main__":
    app.run(debug=True)
