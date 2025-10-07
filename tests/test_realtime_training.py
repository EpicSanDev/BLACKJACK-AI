from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from typing import Dict

from realtime_service.storage import SampleStore
from realtime_service.training import TrainingManager, train_from_samples


def _sample_payload(**overrides: Dict[str, object]) -> Dict[str, object]:
    payload = {
        "client_id": "test-client",
        "player_cards": [
            {"rank": "10", "value": 10},
            {"rank": "6", "value": 6},
        ],
        "dealer_card": {"rank": "9", "value": 9},
        "advisor_action": "Hit",
        "player_action": "Hit",
        "round_outcome": "win",
        "notes": "sample",
    }
    payload.update(overrides)
    return payload


class SampleStoreTests(unittest.TestCase):
    def test_sample_store_persists_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SampleStore(tmpdir)
            image_bytes = base64.b64encode(b"image").decode("ascii")
            payload = _sample_payload(image_base64=image_bytes, image_format="dat")
            record = store.save_sample(payload)

            self.assertIsNotNone(record.image_path)
            stored_path = Path(tmpdir) / record.image_path  # type: ignore[operator]
            self.assertTrue(stored_path.exists())
            self.assertEqual(store.sample_count(), 1)

            stored = list(store.iter_samples())
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["player_cards"][0]["rank"], "10")


class TrainingTests(unittest.TestCase):
    def test_train_from_samples_creates_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SampleStore(tmpdir)
            store.save_sample(_sample_payload())
            store.save_sample(_sample_payload(player_action="Stand"))
            store.save_sample(_sample_payload(player_action="Hit"))

            output_path = Path(tmpdir) / "policy.json"
            result = train_from_samples(store.iter_samples(), output_path)

            self.assertTrue(output_path.exists())
            state_id, state_data = next(iter(result["states"].items()))
            self.assertEqual(state_data["recommended_action"], "Hit")
            self.assertEqual(state_data["action_counts"]["Hit"], 2)
            self.assertEqual(state_data["action_counts"]["Stand"], 1)
            self.assertEqual(result["metadata"]["total_samples"], 3)

    def test_training_manager_runs_background_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SampleStore(tmpdir)
            store.save_sample(_sample_payload())
            output_path = Path(tmpdir) / "policy.json"
            manager = TrainingManager(store, output_path=output_path)

            status = manager.start_training()
            self.assertTrue(status["running"])

            manager._thread.join(timeout=5)  # type: ignore[attr-defined]
            final_status = manager.get_status()
            self.assertFalse(final_status["running"])
            self.assertIsNotNone(final_status.get("last_result"))
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
