"""Model bundle persistence and approved-model registration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib

from junyi_predictor.contracts import ModelRegistration
from junyi_predictor.progress import operation, timed
from junyi_predictor.storage.artifacts import ArtifactStore

logger = logging.getLogger(__name__)


@timed("registration", fields=("model_type",))
def register_model(
    store: ArtifactStore,
    training_run_id: str,
    model_type: str,
    model: object,
    scaler: object,
    metrics: dict[str, dict[str, float]],
) -> ModelRegistration:
    """Persist an immutable model bundle and update the approved-model pointer."""
    version = training_run_id
    prefix = f"models/{version}"
    if store.exists(f"{prefix}/manifest.json"):
        raise ValueError(
            f"training_run_id already has a registered model: {training_run_id}"
        )
    temporary = Path("/tmp") / f"junyi-{training_run_id}"
    temporary.mkdir(parents=True, exist_ok=True)
    model_path = temporary / "model.joblib"
    scaler_path = temporary / "scaler.joblib"
    metrics_path = temporary / "metrics.json"
    with operation("model.serialize", log=logger):
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    model_uri = store.put_file(model_path, f"{prefix}/model.joblib")
    scaler_uri = store.put_file(scaler_path, f"{prefix}/scaler.joblib")
    metrics_uri = store.put_file(metrics_path, f"{prefix}/metrics.json")
    registration = ModelRegistration(
        training_run_id=training_run_id,
        model_version=version,
        model_uri=model_uri,
        scaler_uri=scaler_uri,
        metrics_uri=metrics_uri,
        manifest_uri="",
        model_type=model_type,
        test_score=float(metrics[model_type]["test_score"]),
    )
    manifest_key = f"{prefix}/manifest.json"
    manifest_uri = store.put_json(
        registration.model_dump(mode="json"), f"{prefix}/manifest.json"
    )
    registration = ModelRegistration(
        **{**registration.model_dump(), "manifest_uri": manifest_uri}
    )
    store.put_json(registration.model_dump(mode="json"), manifest_key)
    logger.info(
        "Model manifest persisted",
        extra={"event": "registration.manifest", "key": manifest_key},
    )
    store.put_json(registration.model_dump(mode="json"), "models/approved.json")
    logger.info(
        "Approved model updated",
        extra={
            "event": "registration.approved",
            "model_version": version,
            "test_score": registration.test_score,
        },
    )
    return registration
