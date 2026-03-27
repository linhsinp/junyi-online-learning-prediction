from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from junyi_predictor.pipeline.training import (
    load_feature_matrices,
    load_parquet_dataframe,
    train_and_evaluate_model,
    train_benchmark_model,
)


def test_load_parquet_dataframe_and_feature_matrices(tmp_path):
    parquet_path = tmp_path / "frame.parquet"
    concept_path = tmp_path / "concept.npz"
    level_path = tmp_path / "level.npz"

    source = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    source.to_parquet(parquet_path)
    np.savez_compressed(concept_path, np.array([[1.0], [2.0]]))
    np.savez_compressed(level_path, np.array([[3.0], [4.0]]))

    loaded_frame = load_parquet_dataframe(str(parquet_path))
    concept, level = load_feature_matrices(str(concept_path), str(level_path))

    pd.testing.assert_frame_equal(loaded_frame, source)
    assert concept.shape == (2, 1)
    assert level.shape == (2, 1)


def test_train_and_evaluate_model_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="Unsupported model_type"):
        train_and_evaluate_model(
            X_train=np.array([[0.0], [1.0]]),
            y_train=np.array([0, 1]),
            X_test=np.array([[0.0], [1.0]]),
            y_test=np.array([0, 1]),
            model_type="UnknownModel",
        )


def test_train_benchmark_model_returns_model_and_score(benchmark_frames):
    df_log, df_user, df_content = benchmark_frames
    model, score = train_benchmark_model(
        df_log=df_log, df_user=df_user, df_content=df_content
    )

    assert hasattr(model, "predict")
    assert 0.0 <= score <= 1.0
