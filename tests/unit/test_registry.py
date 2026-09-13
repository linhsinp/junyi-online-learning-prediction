import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from junyi_predictor.contracts import ModelRegistration
from junyi_predictor.registry import register_model
from junyi_predictor.storage.artifacts import LocalArtifactStore


def test_register_model_writes_immutable_bundle_and_approved_pointer(tmp_path: Path):
    model = DecisionTreeClassifier(random_state=0).fit([[0], [1]], [0, 1])
    scaler = MinMaxScaler().fit(np.array([[0.0], [1.0]]))
    store = LocalArtifactStore(tmp_path)

    registration = register_model(
        store=store,
        training_run_id="run-1",
        model_type="DecisionTreeClassifier",
        model=model,
        scaler=scaler,
        metrics={"DecisionTreeClassifier": {"train_score": 1.0, "test_score": 0.9}},
    )

    assert registration.model_version == "run-1"
    assert Path(registration.model_uri).exists()
    assert Path(registration.scaler_uri).exists()
    assert (tmp_path / "models" / "approved.json").exists()
    manifest = json.loads((tmp_path / "models" / "run-1" / "manifest.json").read_text())
    assert manifest["manifest_uri"] == registration.manifest_uri


def test_register_model_rejects_existing_training_run(tmp_path: Path):
    model = DecisionTreeClassifier(random_state=0).fit([[0], [1]], [0, 1])
    scaler = MinMaxScaler().fit(np.array([[0.0], [1.0]]))
    store = LocalArtifactStore(tmp_path)
    kwargs = dict(
        store=store,
        training_run_id="run-1",
        model_type="DecisionTreeClassifier",
        model=model,
        scaler=scaler,
        metrics={"DecisionTreeClassifier": {"train_score": 1.0, "test_score": 0.9}},
    )
    register_model(**kwargs)
    with pytest.raises(ValueError, match="training_run_id"):
        register_model(**kwargs)


def test_old_registration_without_metadata_remains_readable():
    registration = ModelRegistration.model_validate(
        {
            "training_run_id": "old",
            "model_version": "old",
            "model_type": "tree",
            "model_uri": "model",
            "scaler_uri": "scaler",
            "metrics_uri": "metrics",
            "manifest_uri": "manifest",
            "test_score": 0.8,
        }
    )
    assert registration.training_metadata_uri is None


def test_metadata_failure_does_not_publish_or_promote(tmp_path, monkeypatch):
    store = LocalArtifactStore(tmp_path)
    original = store.put_json
    original({"model_version": "old"}, "models/approved.json")

    def fail_metadata(payload, key):
        if key.endswith("training_metadata.json"):
            raise OSError("Synthetic persistence failure")
        return original(payload, key)

    monkeypatch.setattr(store, "put_json", fail_metadata)
    with pytest.raises(OSError):
        register_model(
            store,
            "new",
            "tree",
            {},
            {},
            {"tree": {"test_score": 0.8}},
            training_metadata={"source": "snapshot"},
        )
    assert not (tmp_path / "models/new/manifest.json").exists()
    assert json.loads((tmp_path / "models/approved.json").read_text()) == {
        "model_version": "old"
    }
