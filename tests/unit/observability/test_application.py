from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
from sqlmodel import SQLModel

from junyi_predictor import cli
from junyi_predictor.bootstrap.database import (
    DifficultyEnum,
    to_enum_aware_dict,
    validate_with_sqlmodel,
)
from junyi_predictor.contracts import FeatureSnapshot, PreprocessedSnapshot
from junyi_predictor.pipeline.feature_engineering import build_feature_stage
from junyi_predictor.progress import execution
from junyi_predictor.storage.artifacts import create_artifact_store
from junyi_predictor.workflows import training


@pytest.mark.parametrize(
    "command,function",
    [
        ("download-data", "download_kaggle_data"),
        ("materialize-parquet", "materialize_log_parquet"),
        ("reset-db", "reset_database"),
        ("seed-db", "seed_database_from_raw_files"),
    ],
)
def test_cli_initializes_bootstrap_logging(command, function, records, monkeypatch):
    action = Mock()
    monkeypatch.setattr(cli, function, action)
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setattr("sys.argv", ["junyi-predictor", command])
    cli.main()
    action.assert_called_once()
    assert [r["event"] for r in records()] == [
        "execution.started",
        "execution.completed",
    ]
    assert all(
        r["service"] == "junyi-bootstrap" and r["stage"] == command for r in records()
    )


def test_cli_missing_configuration_is_logged_before_reraise(records, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("sys.argv", ["junyi-predictor", "seed-db"])
    with pytest.raises(KeyError):
        cli.main()
    assert records()[-1]["exception"]["type"] == "KeyError"
    assert records()[-1]["event"] == "execution.failed"


def test_validation_aggregates_reasons_without_input_values(records):
    class Row(SQLModel):
        value: int

    with execution("seed-db", service="junyi-bootstrap"):
        result = validate_with_sqlmodel(
            pd.DataFrame({"value": [1, "learner-secret", "another-private-row"]}), Row
        )
    assert result["value"].tolist() == [1]
    rejected = [r for r in records() if r["event"] == "validation.rejected"]
    assert len(rejected) == 1
    assert rejected[0]["rejected_rows"] == 2
    assert rejected[0]["accepted_rows"] == 1
    assert rejected[0]["reasons"] == {"int_parsing": 2}
    assert "learner-secret" not in str(records())
    assert "another-private-row" not in str(records())


def test_enum_coercion_remains_nullable():
    mapping = {"difficulty": DifficultyEnum}
    assert (
        to_enum_aware_dict({"difficulty": "easy"}, mapping)["difficulty"]
        == DifficultyEnum.easy
    )
    assert to_enum_aware_dict({"difficulty": "unknown"}, mapping)["difficulty"] is None
    assert to_enum_aware_dict({"difficulty": None}, mapping)["difficulty"] is None
    assert (
        to_enum_aware_dict({"difficulty": float("nan")}, mapping)["difficulty"] is None
    )


def test_progress_counters_do_not_change_feature_values(
    records, feature_log_df, feature_user_df, feature_content_df
):
    log = pd.concat([feature_log_df] * 334, ignore_index=True)
    with execution("features"):
        result = build_feature_stage(log, feature_user_df, feature_content_df)
    columns = log["ucid"].map({"c1": 0, "c2": 1}).to_numpy()
    assert np.array_equal(
        result.concept_proficiency[np.arange(len(log)), columns],
        log["level"].to_numpy(),
    )
    assert result.concept_proficiency.shape == (1002, 2)
    completed = [
        r
        for r in records()
        if r["event"] == "operation.completed"
        and r.get("operation") in ("features.concept", "features.level4")
    ]
    assert len(completed) == 2
    assert all(
        r["completed_rows"] == 1002 and r["total_rows"] == 1002 for r in completed
    )


def test_cloud_artifact_events_and_failure_propagation(records, monkeypatch, tmp_path):
    client = MagicMock()
    bucket = client.bucket.return_value
    bucket.name = "test-bucket"
    monkeypatch.setattr(
        "junyi_predictor.storage.artifacts.storage.Client", lambda: client
    )
    with execution("training"):
        store = create_artifact_store("gcs", "prefix", "test-bucket")
        source = tmp_path / "source"
        source.write_text("fixture")
        assert (
            store.put_file(source, "model.bin") == "gs://test-bucket/prefix/model.bin"
        )
        assert (
            store.put_json({"score": 1.0}, "metrics.json")
            == "gs://test-bucket/prefix/metrics.json"
        )
        assert (
            store.get_file("model.bin", tmp_path / "download") == tmp_path / "download"
        )
    assert {r.get("key") for r in records()} >= {"model.bin", "metrics.json"}
    error = OSError("cloud transfer failed")
    bucket.blob.return_value.download_to_filename.side_effect = error
    with pytest.raises(OSError) as caught:
        with execution("training"):
            store.get_file("missing.bin", tmp_path / "missing")
    assert caught.value is error
    assert records()[-1]["operation"] == "artifact.read_file"
    assert records()[-1]["key"] == "missing.bin"


@pytest.mark.parametrize("backend", ["local", "gcs"])
@pytest.mark.parametrize("stage", ["preprocessed", "features"])
def test_snapshot_loading_preserves_local_and_remote_paths(
    backend, stage, records, monkeypatch, tmp_path
):
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("ARTIFACT_BACKEND", backend)
    monkeypatch.setenv("GCS_BUCKET", "fixture")
    monkeypatch.setattr(training.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    store = Mock()
    store.get_file.side_effect = lambda key, destination: destination
    monkeypatch.setattr(training, "_store_from_settings", lambda: store)
    if stage == "preprocessed":
        snapshot = PreprocessedSnapshot(
            training_run_id="r1",
            log_uri="log.parquet",
            user_uri="user.parquet",
            content_uri="content.parquet",
            row_count=5,
        )
        loader = training._load_preprocessed_files
        names = ["log.parquet", "user.parquet", "content.parquet"]
        keys = [f"runs/r1/preprocessed/{name}" for name in names]
    else:
        snapshot = FeatureSnapshot(
            training_run_id="r1",
            log_uri="log.parquet",
            concept_matrix_uri="concept.npy",
            level4_matrix_uri="level4.npy",
            row_count=5,
        )
        loader = training._load_snapshot_files
        names = ["log.parquet", "concept.npy", "level4.npy"]
        keys = [
            "runs/r1/features/log.parquet",
            "runs/r1/features/concept_proficiency.npy",
            "runs/r1/features/level4_proficiency.npy",
        ]
    with execution("training"):
        result = loader(snapshot)
    root = Path(".") if backend == "local" else tmp_path
    assert result == tuple(root / name for name in names)
    assert [call.args[0] for call in store.get_file.call_args_list] == (
        [] if backend == "local" else keys
    )
