"""Utilities for ingesting and persisting client submissions."""

from __future__ import annotations

import base64
import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from blackjack import CARD_VALUES


@dataclass
class SampleRecord:
    """Structured representation of a client submission."""

    sample_id: str
    timestamp: str
    client_id: Optional[str]
    player_cards: List[Dict[str, Any]]
    dealer_card: Dict[str, Any]
    advisor_action: Optional[str]
    player_action: Optional[str]
    round_outcome: Optional[str]
    notes: Optional[str]
    detections: Optional[List[Dict[str, Any]]]
    image_path: Optional[str]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class SampleStore:
    """Filesystem-backed storage for realtime gameplay samples."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = self.base_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.base_dir / "samples.jsonl"
        self._lock = threading.Lock()
        self._sample_count = self._initial_count()

    def _initial_count(self) -> int:
        if not self.samples_path.exists():
            return 0
        with self.samples_path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def save_sample(self, payload: Mapping[str, Any]) -> SampleRecord:
        """Validate, persist and return a structured submission."""

        record = self._build_record(payload)
        with self._lock:
            with self.samples_path.open("a", encoding="utf-8") as handle:
                handle.write(record.to_json())
                handle.write("\n")
            self._sample_count += 1
        return record

    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """Yield stored submissions as dictionaries."""

        if not self.samples_path.exists():
            return
        with self.samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def sample_count(self) -> int:
        """Return the number of stored samples."""

        return self._sample_count

    def _build_record(self, payload: Mapping[str, Any]) -> SampleRecord:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        client_id = str(payload.get("client_id") or "anonymous")
        player_cards = self._normalise_cards(payload.get("player_cards"), allow_zero=True)
        dealer_card_list = self._normalise_cards([payload.get("dealer_card")], allow_zero=False)
        dealer_card = dealer_card_list[0]

        advisor_action = self._normalise_action(payload.get("advisor_action"))
        player_action = self._normalise_action(payload.get("player_action"))
        round_outcome = self._normalise_text(payload.get("round_outcome"))
        notes = self._normalise_text(payload.get("notes"))
        detections = self._normalise_detections(payload.get("detections"))
        image_path = self._store_image(payload)

        return SampleRecord(
            sample_id=str(uuid.uuid4()),
            timestamp=timestamp,
            client_id=client_id,
            player_cards=player_cards,
            dealer_card=dealer_card,
            advisor_action=advisor_action,
            player_action=player_action,
            round_outcome=round_outcome,
            notes=notes,
            detections=detections,
            image_path=image_path,
        )

    def _normalise_cards(
        self,
        cards: Optional[Iterable[Mapping[str, Any]]],
        *,
        allow_zero: bool,
    ) -> List[Dict[str, Any]]:
        if not cards:
            raise ValueError("Card information is required")

        normalised: List[Dict[str, Any]] = []
        for card in cards:
            if not card:
                continue
            rank = str(card.get("rank", "")).lower().strip()
            value = card.get("value")
            if value is None:
                if rank not in CARD_VALUES:
                    raise ValueError(f"Unknown card rank '{rank}' and no value provided")
                value = CARD_VALUES[rank]
            value = int(value)
            if value == 0 and not allow_zero:
                raise ValueError("Dealer card must have a non-zero value")
            normalised.append({"rank": rank, "value": value})

        if not normalised:
            raise ValueError("At least one card must be provided")
        return normalised

    @staticmethod
    def _normalise_action(action: Any) -> Optional[str]:
        if action is None:
            return None
        text = str(action).strip()
        return text or None

    @staticmethod
    def _normalise_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _normalise_detections(detections: Any) -> Optional[List[Dict[str, Any]]]:
        if not detections:
            return None
        normalised: List[Dict[str, Any]] = []
        for detection in detections:
            if not isinstance(detection, Mapping):
                continue
            entry: Dict[str, Any] = {
                "rank": str(detection.get("rank", "")).lower(),
                "value": int(detection.get("value", 0)),
            }
            if "confidence" in detection:
                entry["confidence"] = float(detection["confidence"])
            if "box" in detection:
                entry["box"] = [float(v) for v in detection["box"]]
            normalised.append(entry)
        return normalised or None

    def _store_image(self, payload: Mapping[str, Any]) -> Optional[str]:
        image_b64 = payload.get("image_base64")
        if not image_b64:
            return None
        image_format = str(payload.get("image_format") or "png").strip(". ") or "png"
        sample_id = uuid.uuid4().hex
        filename = f"{sample_id}.{image_format}"
        target_path = self.upload_dir / filename

        data = base64.b64decode(image_b64)
        with target_path.open("wb") as handle:
            handle.write(data)
        try:
            relative = target_path.relative_to(self.base_dir)
            return str(relative)
        except ValueError:  # pragma: no cover - defensive safety
            return str(target_path)


__all__ = ["SampleStore", "SampleRecord"]
