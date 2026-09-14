from __future__ import annotations

import pandas as pd

from junyi_predictor.bootstrap.database import (
    InfoContent,
    UserProfile,
    reset_database,
    seed_database_from_gcs,
    seed_database_from_raw_files,
)
from junyi_predictor.paths import CONTENT_FILE, USER_FILE


def test_seed_database_from_raw_files_seeds_only_dimensions(monkeypatch):
    frames = {
        USER_FILE: pd.DataFrame(
            {"uuid": ["u1"], "first_login_date_TW": ["2024-01-01"]}
        ),
        CONTENT_FILE: pd.DataFrame({"ucid": ["c1"]}),
    }
    read_paths = []
    created_tables: list[tuple[str, str]] = []

    def fake_read_csv(path, **_kwargs):
        read_paths.append(path)
        return frames[path].copy()

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        "junyi_predictor.bootstrap.database.create_engine", lambda engine_url: "engine"
    )
    monkeypatch.setattr(
        "junyi_predictor.bootstrap.database.create_table_from_dataframe",
        lambda df, model_class, engine: created_tables.append(
            (model_class.__name__, engine)
        ),
    )
    seed_database_from_raw_files(engine_url="postgresql://example")

    assert read_paths == [USER_FILE, CONTENT_FILE]
    assert created_tables == [("InfoContent", "engine"), ("UserProfile", "engine")]


def test_seed_database_exports_expected_sqlmodel_types():
    assert InfoContent.__tablename__ == "info_content"
    assert UserProfile.__tablename__ == "user_profile"


def test_seed_database_from_gcs_downloads_staged_dimension_files(monkeypatch):
    downloaded: list[tuple[str, object]] = []
    bucket = type("Bucket", (), {})()

    class Blob:
        def __init__(self, key):
            self.key = key

        def download_to_filename(self, destination):
            downloaded.append((self.key, destination))

    bucket.blob = lambda key: Blob(key)
    client = type("Client", (), {"bucket": lambda self, _name: bucket})()
    seeded: list[tuple[object, object, str]] = []
    monkeypatch.setattr(
        "junyi_predictor.bootstrap.database.seed_database_from_csv_paths",
        lambda user, content, engine: seeded.append((user, content, engine)),
    )

    seed_database_from_gcs(
        "lake", "data/dimensions", "postgresql://example", client=client
    )

    assert [key for key, _destination in downloaded] == [
        "data/dimensions/Info_UserData.csv",
        "data/dimensions/Info_Content.csv",
    ]
    assert seeded[0][2] == "postgresql://example"


def test_reset_database_removes_source_and_workflow_tables_before_dimensions(
    monkeypatch,
):
    statements: list[str] = []
    dropped_engines: list[object] = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement):
            statements.append(str(statement))

    class Engine:
        def begin(self):
            return Connection()

    engine = Engine()
    monkeypatch.setattr(
        "junyi_predictor.bootstrap.database.create_engine", lambda _url: engine
    )
    monkeypatch.setattr(
        "junyi_predictor.bootstrap.database.SQLModel.metadata.drop_all",
        lambda received_engine: dropped_engines.append(received_engine),
    )

    reset_database("postgresql://example")

    assert statements == [
        "DROP TABLE IF EXISTS log_problem CASCADE",
        "DROP TABLE IF EXISTS processed_log CASCADE",
        "DROP TABLE IF EXISTS feature_snapshot CASCADE",
    ]
    assert dropped_engines == [engine]
