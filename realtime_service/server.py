"""Flask application exposing APIs for realtime data collection and training."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

from .storage import SampleStore
from .training import TrainingManager


def create_app(config: Dict[str, Any] | None = None) -> Flask:
    """Create and configure the realtime training server."""

    config = config or {}
    data_dir = Path(config.get("data_dir") or os.environ.get("BLACKJACK_SERVER_DATA", "server_data"))
    policy_path = config.get("policy_path") or os.environ.get("BLACKJACK_POLICY_PATH")

    store = SampleStore(data_dir)
    trainer = TrainingManager(store, output_path=policy_path)

    app = Flask(__name__)
    app.config["SAMPLE_STORE"] = store
    app.config["TRAINING_MANAGER"] = trainer

    @app.get("/health")
    def health() -> Any:
        return {"status": "ok"}

    @app.get("/api/v1/status")
    def status() -> Any:
        trainer_status = trainer.get_status()
        return jsonify(
            {
                "samples": {
                    "count": store.sample_count(),
                    "data_dir": str(store.base_dir),
                },
                "training": trainer_status,
                "policy_path": str(trainer.output_path),
            }
        )

    @app.post("/api/v1/samples")
    def submit_sample() -> Any:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"error": "Invalid JSON payload"}), 400
        try:
            record = store.save_sample(payload)
        except ValueError as exc:
            logging.exception("Failed to save sample")
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.exception("Unexpected error while saving sample")
            return jsonify({"error": str(exc)}), 500

        response = {
            "sample_id": record.sample_id,
            "timestamp": record.timestamp,
            "stored_image": record.image_path,
        }
        return jsonify(response), 201

    @app.post("/api/v1/train")
    def trigger_training() -> Any:
        try:
            status = trainer.start_training()
        except RuntimeError as exc:
            return jsonify({"error": str(exc), "status": trainer.get_status()}), 409
        return jsonify({"message": "Training started", "status": status})

    @app.get("/api/v1/policy")
    def get_policy() -> Any:
        policy = trainer.current_policy()
        if not policy:
            return jsonify({"error": "No policy trained yet"}), 404
        return jsonify(policy)

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("BLACKJACK_SERVER_PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
