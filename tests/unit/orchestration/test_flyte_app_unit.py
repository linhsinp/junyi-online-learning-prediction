from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from junyi_predictor.pipeline.feature_engineering import FeatureStageOutput
from junyi_predictor.pipeline.preprocessing import PreprocessStageOutput
from orchestration import flyte_app


def test_train_all_models_covers_all_configured_model_types(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def fake_train_and_evaluate_model(**kwargs):
        calls.append(kwargs["model_type"])
        return {"train_score": 0.9, "test_score": 0.8}

    monkeypatch.setattr(
        flyte_app, "train_and_evaluate_model", fake_train_and_evaluate_model
    )

    metrics = flyte_app._train_all_models(
        X_train=np.array([[0.0], [1.0]]),
        y_train=np.array([0, 1]),
        X_test=np.array([[0.0], [1.0]]),
        y_test=np.array([0, 1]),
    )

    assert calls == list(flyte_app.MODEL_TYPES)
    assert set(metrics) == set(flyte_app.MODEL_TYPES)


@pytest.mark.anyio
async def test_preprocess_from_database_returns_summary(
    monkeypatch: pytest.MonkeyPatch, mock_stage_frames
):
    df_log, df_user, df_content = mock_stage_frames

    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(flyte_app, "create_engine", lambda url: f"engine:{url}")
    monkeypatch.setattr(
        flyte_app,
        "load_data_from_database",
        lambda **kwargs: (df_log, df_user, df_content),
    )
    monkeypatch.setattr(
        flyte_app,
        "preprocess_stage",
        lambda **kwargs: PreprocessStageOutput(
            log=df_log, user=df_user, content=df_content
        ),
    )

    result = await flyte_app.preprocess_from_database.func(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
    )

    assert result == {
        "log_rows": 3,
        "user_rows": 2,
        "content_rows": 1,
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-01-02T00:00:00",
    }


@pytest.mark.anyio
async def test_full_pipeline_uses_stage_contracts(
    monkeypatch: pytest.MonkeyPatch, mock_stage_frames, mocked_stage_split
):
    df_log, df_user, df_content = mock_stage_frames

    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(flyte_app, "create_engine", lambda url: f"engine:{url}")
    monkeypatch.setattr(
        flyte_app,
        "load_data_from_database",
        lambda **kwargs: (df_log, df_user, df_content),
    )
    monkeypatch.setattr(
        flyte_app,
        "preprocess_stage",
        lambda **kwargs: PreprocessStageOutput(
            log=df_log, user=df_user, content=df_content
        ),
    )
    monkeypatch.setattr(
        flyte_app,
        "build_feature_stage",
        lambda **kwargs: FeatureStageOutput(
            log=df_log,
            concept_proficiency=np.ones((3, 1)),
            level4_proficiency=np.ones((3, 1)),
        ),
    )
    monkeypatch.setattr(
        flyte_app, "split_training_data", lambda **kwargs: mocked_stage_split
    )
    monkeypatch.setattr(
        flyte_app,
        "apply_min_max_transformation",
        lambda X_train, X_test: (X_train, X_test),
    )
    monkeypatch.setattr(
        flyte_app,
        "_train_all_models",
        lambda X_train, y_train, X_test, y_test: {
            "DecisionTreeClassifier": {"test_score": 1.0}
        },
    )

    result = await flyte_app.full_pipeline.func(num_samples=3)

    assert result == {"DecisionTreeClassifier": {"test_score": 1.0}}


@pytest.mark.anyio
async def test_train_from_gcs_downloads_features_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path, mocked_stage_split
):
    local_dir = tmp_path / "gcs"
    removed: list[str] = []

    monkeypatch.setattr(flyte_app, "download_data_to_tmp", lambda **kwargs: None)
    monkeypatch.setattr(
        flyte_app,
        "load_parquet_dataframe",
        lambda path: pd.DataFrame({"loaded_from": [path]}),
    )
    monkeypatch.setattr(
        flyte_app,
        "load_feature_matrices",
        lambda *args: (np.ones((2, 1)), np.ones((2, 1))),
    )
    monkeypatch.setattr(
        flyte_app, "split_training_data", lambda **kwargs: mocked_stage_split
    )
    monkeypatch.setattr(
        flyte_app,
        "apply_min_max_transformation",
        lambda X_train, X_test: (X_train, X_test),
    )
    monkeypatch.setattr(
        flyte_app,
        "_train_all_models",
        lambda X_train, y_train, X_test, y_test: {
            "GradientBoostingClassifier": {"test_score": 0.9}
        },
    )
    monkeypatch.setattr(
        flyte_app.shutil,
        "rmtree",
        lambda path, ignore_errors: removed.append(path),
    )

    result = await flyte_app.train_from_gcs.func(local_dir=str(local_dir))

    assert result == {"GradientBoostingClassifier": {"test_score": 0.9}}
    assert removed == [str(local_dir)]
