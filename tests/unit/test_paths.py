from __future__ import annotations

from junyi_predictor.paths import (
    ARTIFACTS_DIR,
    CONTENT_FILE,
    DATA_DIR,
    FEATURE_STORE_DIR,
    LOG_FILE,
    MODEL_DIR,
    RAW_DATA_DIR,
    USER_FILE,
)


def test_artifact_paths_are_rooted_under_artifacts():
    assert ARTIFACTS_DIR.as_posix() == "artifacts"
    assert DATA_DIR.as_posix() == "artifacts/data"
    assert MODEL_DIR.as_posix() == "artifacts/model"
    assert RAW_DATA_DIR.as_posix() == "artifacts/data/raw"
    assert FEATURE_STORE_DIR.as_posix() == "artifacts/data/feature_store"
    assert LOG_FILE.as_posix() == "artifacts/data/raw/Log_Problem.csv"
    assert USER_FILE.as_posix() == "artifacts/data/raw/Info_UserData.csv"
    assert CONTENT_FILE.as_posix() == "artifacts/data/raw/Info_Content.csv"
