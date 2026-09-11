"""Model bundle persistence and approved-model registration."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from junyi_predictor.contracts import ModelRegistration
from junyi_predictor.storage.artifacts import ArtifactStore


def register_model(
    store: ArtifactStore,
    run_id: str,
    model_type: str,
    model: object,
    scaler: object,
    metrics: dict[str, dict[str, float]],
) -> ModelRegistration:
    """Persist an immutable model bundle and update the approved-model pointer."""
    version = run_id
    temporary = Path("/tmp") / f"junyi-{run_id}"
    temporary.mkdir(parents=True, exist_ok=True)
    model_path = temporary / "model.joblib"
    scaler_path = temporary / "scaler.joblib"
    metrics_path = temporary / "metrics.json"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    prefix = f"models/{version}"
    model_uri = store.put_file(model_path, f"{prefix}/model.joblib")
    scaler_uri = store.put_file(scaler_path, f"{prefix}/scaler.joblib")
    metrics_uri = store.put_file(metrics_path, f"{prefix}/metrics.json")
    registration = ModelRegistration(
        run_id=run_id,
        model_version=version,
        model_uri=model_uri,
        scaler_uri=scaler_uri,
        metrics_uri=metrics_uri,
        manifest_uri="",
        model_type=model_type,
        test_score=float(metrics[model_type]["test_score"]),
    )
    manifest_uri = store.put_json(registration.to_dict(), f"{prefix}/manifest.json")
    registration = ModelRegistration(
        **{**registration.to_dict(), "manifest_uri": manifest_uri}
    )
    store.put_json(registration.to_dict(), "models/approved.json")
    return registration
