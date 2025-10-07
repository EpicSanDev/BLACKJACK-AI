"""Reusable Blackjack card detection helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from utils import select_best_device

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


@dataclass
class CardDetection:
    """Represents a single detected card in an image."""

    rank: str
    value: int
    box: Tuple[float, float, float, float]
    confidence: float
    center: Tuple[float, float]

    def to_strategy_card(self) -> Dict[str, int]:
        """Return a simplified mapping compatible with strategy helpers."""

        return {"rank": self.rank, "value": int(self.value)}

    def to_payload(self) -> Dict[str, object]:
        """Return a serialisable representation for API payloads."""

        return {
            "rank": self.rank,
            "value": int(self.value),
            "box": tuple(float(coord) for coord in self.box),
            "confidence": float(self.confidence),
            "center": tuple(float(c) for c in self.center),
        }


class BlackjackDetector:
    """Wrapper around a YOLO model to detect blackjack cards."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        imgsz: int | None = None,
        card_value_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")

        self.card_value_map = dict(card_value_map or CARD_VALUE_MAP)
        requested_device = device or os.environ.get("YOLO_DEVICE")
        self.device = select_best_device(requested_device)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = imgsz

        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

    def detect(self, image_path: Path | str) -> List[CardDetection]:
        """Run the YOLO model and return all detected cards."""

        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
        )
        detections: List[CardDetection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[int(cls_id)]
                rank = class_name.split("_")[0]
                value = int(self.card_value_map.get(rank, 0))
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append(
                    CardDetection(
                        rank=rank,
                        value=value,
                        box=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(box.conf[0]),
                        center=((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2),
                    )
                )
        return detections

    def detect_from_image(self, image: np.ndarray) -> List[CardDetection]:
        """Run the YOLO model on an in-memory image and return all detected cards."""

        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
        )
        detections: List[CardDetection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[int(cls_id)]
                rank = class_name.split("_")[0]
                value = int(self.card_value_map.get(rank, 0))
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append(
                    CardDetection(
                        rank=rank,
                        value=value,
                        box=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(box.conf[0]),
                        center=((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2),
                    )
                )
        return detections

    @staticmethod
    def group_cards(cards: Sequence[CardDetection]) -> Tuple[List[CardDetection], Optional[CardDetection]]:
        """Split detections into player cards and dealer up-card."""

        if not cards:
            return [], None

        cards_sorted = sorted(cards, key=lambda c: c.center[1], reverse=True)
        player_cards = list(cards_sorted[:2])
        dealer_candidates = [card for card in cards_sorted[2:] if card.value > 0] or cards_sorted[1:]
        dealer_card = min(dealer_candidates, key=lambda c: c.center[1]) if dealer_candidates else None
        return player_cards, dealer_card

    @staticmethod
    def annotate_image(
        image_path: Path | str,
        detections: Sequence[CardDetection],
        output_dir: Path,
    ) -> Path:
        """Return a path to an annotated copy of the analysed image."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(image_path))
        for card in detections:
            x1, y1, x2, y2 = map(int, card.box)
            label = f"{card.rank} ({card.confidence:.2f})"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
            )

        annotated_path = output_dir / f"annotated_{Path(image_path).name}"
        cv2.imwrite(str(annotated_path), image)
        return annotated_path

    @staticmethod
    def serialize_detections(detections: Iterable[CardDetection]) -> List[Dict[str, object]]:
        """Serialise detections into dictionaries."""

        return [card.to_payload() for card in detections]


__all__ = ["BlackjackDetector", "CardDetection", "CARD_VALUE_MAP"]
