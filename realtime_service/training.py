"""Training utilities for aggregating crowdsourced blackjack data."""

from __future__ import annotations

import json
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

from blackjack import ACTION_LABELS, normalise_hand, evaluate_hand


ActionCounter = Counter


def _state_key(total: int, is_soft: bool, dealer_value: int) -> str:
    kind = "soft" if is_soft else "hard"
    return f"{total}:{kind}:{dealer_value}"


def train_from_samples(
    samples: Iterable[Mapping[str, object]],
    output_path: Path | str,
) -> Dict[str, object]:
    """Aggregate samples and write a majority-vote policy to ``output_path``."""

    action_space = set(ACTION_LABELS.keys())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_counters: MutableMapping[str, ActionCounter] = defaultdict(ActionCounter)
    total_samples = 0
    skipped_samples = 0

    for sample in samples:
        player_cards = sample.get("player_cards")
        dealer_card = sample.get("dealer_card")
        action = sample.get("player_action") or sample.get("advisor_action")
        if not player_cards or not dealer_card or not action:
            skipped_samples += 1
            continue

        try:
            normalised_player = normalise_hand(player_cards)  # type: ignore[arg-type]
            dealer_normalised = normalise_hand([dealer_card])  # type: ignore[arg-type]
            total, is_soft = evaluate_hand(normalised_player)
            dealer_value = int(dealer_normalised[0]["value"])
        except Exception:
            skipped_samples += 1
            continue

        action_name = str(action)
        if action_name not in action_space:
            skipped_samples += 1
            continue

        key = _state_key(int(total), bool(is_soft), dealer_value)
        state_counters[key][action_name] += 1
        total_samples += 1

    states: Dict[str, Dict[str, object]] = {}
    for key, counter in state_counters.items():
        if not counter:
            continue
        best_action, best_count = counter.most_common(1)[0]
        total_state_samples = int(sum(counter.values()))
        states[key] = {
            "recommended_action": best_action,
            "confidence": round(best_count / total_state_samples, 4),
            "total_samples": total_state_samples,
            "action_counts": {action: int(count) for action, count in counter.items()},
        }

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_samples": total_samples,
            "skipped_samples": skipped_samples,
            "unique_states": len(states),
        },
        "states": states,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return payload


class TrainingManager:
    """Coordinates background training jobs for the realtime server."""

    def __init__(self, store, output_path: Path | str | None = None) -> None:
        self.store = store
        self.output_path = Path(output_path or Path("model") / "community_policy.json")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._status: Dict[str, object] = {
            "running": False,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "last_result": None,
        }
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start_training(self) -> Dict[str, object]:
        with self._lock:
            if self._status["running"]:
                raise RuntimeError("Training already in progress")
            self._status.update(
                running=True,
                started_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                completed_at=None,
                error=None,
            )
            self._thread = threading.Thread(target=self._run_training, daemon=True)
            self._thread.start()
            return dict(self._status)

    def _run_training(self) -> None:
        try:
            result = train_from_samples(self.store.iter_samples(), self.output_path)
            status_update = {
                "running": False,
                "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "last_result": result,
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            status_update = {
                "running": False,
                "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "error": str(exc),
                "last_result": None,
            }
        with self._lock:
            self._status.update(status_update)

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._status)

    def current_policy(self) -> Optional[Dict[str, object]]:
        if not self.output_path.exists():
            return None
        with self.output_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


__all__ = ["TrainingManager", "train_from_samples"]
