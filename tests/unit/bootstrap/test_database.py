from __future__ import annotations

import pandas as pd

from junyi_predictor.bootstrap.database import (
    InfoContent,
    LogProblem,
    UserProfile,
    seed_database_from_raw_files,
)
from junyi_predictor.paths import CONTENT_FILE, LOG_FILE, USER_FILE


def test_seed_database_from_raw_files_uses_artifact_raw_paths(monkeypatch):
    frames = (
        pd.DataFrame({"timestamp_TW": ["2024-01-01"], "uuid": ["u1"]}),
        pd.DataFrame({"uuid": ["u1"], "first_login_date_TW": ["2024-01-01"]}),
        pd.DataFrame({"ucid": ["c1"]}),
    )
    captured_paths: list[tuple[str, str, str]] = []
    created_tables: list[tuple[str, str]] = []
    chunk_calls: list[tuple[str, str]] = []

    def fake_load_raw_dataframes(path_log: str, path_user: str, path_content: str):
        captured_paths.append((path_log, path_user, path_content))
        return frames

    monkeypatch.setattr(
        "junyi_predictor.bootstrap.database.load_raw_dataframes",
        fake_load_raw_dataframes,
    )
    monkeypatch.setattr(
        "junyi_predictor.bootstrap.database.create_engine", lambda engine_url: "engine"
    )
    monkeypatch.setattr(
        "junyi_predictor.bootstrap.database.create_table_from_dataframe",
        lambda df, model_class, engine: created_tables.append(
            (model_class.__name__, engine)
        ),
    )
    monkeypatch.setattr(
        "junyi_predictor.bootstrap.database.chunked_upload_with_validation",
        lambda df, model_class, engine, table_name: chunk_calls.append(
            (model_class.__name__, table_name)
        ),
    )

    seed_database_from_raw_files(engine_url="postgresql://example")

    assert captured_paths == [(str(LOG_FILE), str(USER_FILE), str(CONTENT_FILE))]
    assert created_tables == [("InfoContent", "engine"), ("UserProfile", "engine")]
    assert chunk_calls == [("LogProblem", "log_problem")]


def test_seed_database_exports_expected_sqlmodel_types():
    assert InfoContent.__tablename__ == "info_content"
    assert UserProfile.__tablename__ == "user_profile"
    assert LogProblem.__tablename__ == "log_problem"
