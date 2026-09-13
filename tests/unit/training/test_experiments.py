"""Regression coverage for reusable features and evaluation isolation."""

import json
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from junyi_predictor.pipeline import experiments
from junyi_predictor.pipeline.experiments import select_and_split, train_snapshot
from junyi_predictor.settings import ArtifactSettings, Settings


def test_default_split_selects_all_rows_and_preserves_matrix_alignment(
    experiment_frames,
):
    split = select_and_split(*experiment_frames, expected_rows=20)
    assert [len(split.y_train), len(split.y_validation), len(split.y_test)] == [
        14,
        3,
        3,
    ]
    combined = np.concatenate([split.X_train, split.X_validation, split.X_test])
    assert combined[:, 0].tolist() == list(range(20))
    assert combined[:, -2].tolist() == list(range(20))
    assert combined[:, -1].tolist() == list(range(100, 120))
    assert split.metadata["selected_row_count"] == 20
    assert split.metadata["partitions"]["test"]["start_position"] == 17


def test_date_filter_precedes_configurable_split(experiment_frames):
    split = select_and_split(
        *experiment_frames,
        expected_rows=20,
        start_date=datetime.fromisoformat("2024-01-03T02:00:00+02:00"),
        end_date=datetime(2024, 1, 13),
        train_fraction=0.6,
        validation_fraction=0.2,
    )
    assert [len(split.y_train), len(split.y_validation), len(split.y_test)] == [6, 2, 2]
    combined = np.concatenate([split.X_train, split.X_validation, split.X_test])
    assert combined[:, 0].tolist() == list(range(2, 12))
    assert combined[:, -2].tolist() == list(range(2, 12))
    assert split.metadata["requested_start_date"] == "2024-01-03T00:00:00+00:00"


def test_equal_timestamps_stay_in_one_partition(experiment_frames):
    log, concept, level4 = experiment_frames
    log.loc[14, "timestamp_TW"] = log.loc[13, "timestamp_TW"]
    log.loc[17, "timestamp_TW"] = log.loc[16, "timestamp_TW"]
    split = select_and_split(log, concept, level4, expected_rows=20)
    assert [len(split.y_train), len(split.y_validation), len(split.y_test)] == [
        15,
        3,
        2,
    ]


@pytest.mark.parametrize(
    "train,validation",
    [(0, 0.15), (0.7, 0), (0.9, 0.1), (float("nan"), 0.1), (0.7, float("inf"))],
)
def test_invalid_fractions_rejected(experiment_frames, train, validation):
    with pytest.raises(ValueError, match="fractions"):
        select_and_split(
            *experiment_frames,
            expected_rows=20,
            train_fraction=train,
            validation_fraction=validation,
        )


@pytest.mark.parametrize(
    "problem",
    [
        "missing_column",
        "row_count",
        "matrix_rows",
        "matrix_dimension",
        "matrix_nan",
        "matrix_string",
        "scalar_nan",
        "scalar_string",
        "target_null",
        "target_invalid",
        "timestamp_null",
        "unsorted",
        "one_class",
        "all_tied",
    ],
)
def test_invalid_inputs_fail_before_fitting(experiment_frames, problem):
    log, concept, level4 = experiment_frames
    expected_rows = 20
    if problem == "missing_column":
        log = log.drop(columns="user_grade")
    elif problem == "row_count":
        expected_rows = 21
    elif problem == "matrix_rows":
        concept = concept[:-1]
    elif problem == "matrix_dimension":
        concept = concept[:, 0]
    elif problem == "matrix_nan":
        concept[0, 0] = np.nan
    elif problem == "matrix_string":
        concept = concept.astype(str)
    elif problem == "scalar_nan":
        log["user_grade"] = float("nan")
    elif problem == "scalar_string":
        log["user_grade"] = "not numeric"
    elif problem == "target_null":
        log["is_correct"] = None
    elif problem == "target_invalid":
        log["is_correct"] = 2
    elif problem == "timestamp_null":
        log["timestamp_TW"] = pd.NaT
    elif problem == "unsorted":
        log = log.iloc[::-1]
    elif problem == "one_class":
        log["is_correct"] = True
    elif problem == "all_tied":
        log["timestamp_TW"] = pd.Timestamp("2024-01-01", tz="UTC")
    with pytest.raises(ValueError):
        select_and_split(log, concept, level4, expected_rows=expected_rows)


@pytest.mark.parametrize(
    "start,end",
    [
        (datetime(2025, 1, 1), None),
        (datetime(2024, 1, 1), datetime(2024, 1, 3)),
        (datetime(2024, 1, 3), datetime(2024, 1, 1)),
        (pd.NaT, None),
    ],
)
def test_invalid_or_insufficient_date_selection(experiment_frames, start, end):
    with pytest.raises(ValueError):
        select_and_split(
            *experiment_frames, expected_rows=20, start_date=start, end_date=end
        )


def test_model_selection_and_test_are_isolated(experiment_frames, monkeypatch):
    split = select_and_split(*experiment_frames, expected_rows=20)
    scaler = Mock()
    scaler.transform.side_effect = lambda values: values
    fit_scaler = Mock(return_value=scaler)
    monkeypatch.setattr(experiments, "fit_min_max_scaler", fit_scaler)
    # B would win on test; A wins validation, ties C, and must be the only model tested.
    models = {name: Mock() for name in ("A", "B", "C")}
    for name, model in models.items():
        model.get_params.return_value = {"random_state": 0}
        model.score.side_effect = [
            1.0,
            0.9 if name != "B" else 0.8,
            0.1 if name == "A" else 1.0,
        ]
    fit = Mock(side_effect=lambda X, y, name: models[name])
    monkeypatch.setattr(experiments, "MODEL_TYPES", tuple(models))
    monkeypatch.setattr(experiments, "fit_model", fit)
    model, _, winner, metrics, _ = experiments.evaluate_candidates(split)
    assert winner == "A" and model is models["A"]
    assert metrics["A"]["test_score"] == 0.1
    assert "test_score" not in metrics["B"] and "test_score" not in metrics["C"]
    assert fit_scaler.call_args.args[0] is split.X_train
    assert all(
        call.args[0] is split.X_train and call.args[1] is split.y_train
        for call in fit.call_args_list
    )
    assert models["A"].score.call_args_list[-1].args[0] is split.X_test
    assert models["B"].score.call_count == models["C"].score.call_count == 2
    assert models["B"].score.call_args_list[-1].args[0] is split.X_validation


def test_actual_scaler_uses_only_training_range(experiment_frames):
    split = select_and_split(*experiment_frames, expected_rows=20)
    _, scaler, _, _, _ = experiments.evaluate_candidates(split)
    assert scaler.data_max_[0] == 13
    assert scaler.transform(split.X_test)[0, 0] > 1


def test_training_registers_metadata_without_promoting(
    published_feature_snapshot, monkeypatch
):
    store, key = published_feature_snapshot
    monkeypatch.delenv("DATABASE_URL", raising=False)
    approved = store.root / "models/approved.json"
    store.put_json({"model_version": "existing"}, "models/approved.json")
    previous = approved.read_bytes()
    registration = train_snapshot(store, key, "experiment")
    assert approved.read_bytes() == previous
    metadata = json.loads(
        (store.root / "models/experiment/training_metadata.json").read_text()
    )
    assert registration.training_metadata_uri
    assert metadata["source_training_run_id"] == "source-run"
    assert metadata["training_run_id"] == "experiment"
    assert len(metadata["input_sha256"]) == 4
    assert metadata["selection"]["selected_row_count"] == 20
    assert len(metadata["candidate_configurations"]) == 4
    metrics = json.loads((store.root / "models/experiment/metrics.json").read_text())
    assert sum("test_score" in item for item in metrics.values()) == 1


def test_registered_run_rejected_before_loading_or_fitting(
    published_feature_snapshot, monkeypatch
):
    store, key = published_feature_snapshot
    store.put_json({}, "models/existing/manifest.json")
    load = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr(experiments, "load_feature_snapshot", load)
    with pytest.raises(ValueError, match="already"):
        train_snapshot(store, key, "existing")
    load.assert_not_called()


@pytest.mark.parametrize("run_id", ["", "../other", "/absolute", "a/b"])
def test_training_run_id_cannot_escape_artifact_directory(
    published_feature_snapshot, run_id
):
    store, key = published_feature_snapshot
    with pytest.raises(ValueError, match="training_run_id"):
        train_snapshot(store, key, run_id)


def test_artifact_settings_do_not_require_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert ArtifactSettings.from_environment().artifact_backend
    with pytest.raises(KeyError):
        Settings.from_environment()
