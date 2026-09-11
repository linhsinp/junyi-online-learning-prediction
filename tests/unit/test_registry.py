import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from junyi_predictor.registry import register_model
from junyi_predictor.storage.artifacts import LocalArtifactStore


def test_register_model_writes_immutable_bundle_and_approved_pointer(tmp_path: Path):
    model = DecisionTreeClassifier(random_state=0).fit([[0], [1]], [0, 1])
    scaler = MinMaxScaler().fit(np.array([[0.0], [1.0]]))
    store = LocalArtifactStore(tmp_path)

    registration = register_model(
        store=store,
        run_id="run-1",
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
