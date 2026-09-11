from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from junyi_predictor.pipeline.preprocessing import (
    load_data_for_training,
    load_raw_dataframes,
)


def test_load_raw_dataframes_reads_csv_inputs_with_expected_shapes(tmp_path):
    log_path = tmp_path / "log.csv"
    user_path = tmp_path / "user.csv"
    content_path = tmp_path / "content.csv"

    pd.DataFrame(
        {
            "timestamp_TW": ["2024-01-01"],
            "uuid": ["u1"],
            "ucid": ["c1"],
            "upid": ["p1"],
            "problem_number": [1],
            "exercise_problem_repeat_session": [0],
            "is_correct": [True],
            "total_sec_taken": [10],
            "total_attempt_cnt": [1],
            "used_hint_cnt": [0],
            "is_hint_used": [False],
            "level": [1],
        }
    ).to_csv(log_path, index=False)
    pd.DataFrame({"uuid": ["u1"], "gender": ["female"], "user_grade": [5]}).to_csv(
        user_path, index=False
    )
    pd.DataFrame(
        {
            "ucid": ["c1"],
            "level4_id": ["l1"],
            "difficulty": ["easy"],
            "learning_stage": ["elementary"],
        }
    ).to_csv(content_path, index=False)

    df_log, df_user, df_content = load_raw_dataframes(
        str(log_path), str(user_path), str(content_path)
    )

    assert df_log.shape == (1, 12)
    assert df_user.shape == (1, 3)
    assert df_content.shape == (1, 4)


def test_load_raw_dataframes_rejects_empty_inputs(tmp_path):
    log_path = tmp_path / "log.csv"
    user_path = tmp_path / "user.csv"
    content_path = tmp_path / "content.csv"

    pd.DataFrame(columns=["timestamp_TW"]).to_csv(log_path, index=False)
    pd.DataFrame({"uuid": ["u1"], "gender": ["female"], "user_grade": [5]}).to_csv(
        user_path, index=False
    )
    pd.DataFrame(
        {
            "ucid": ["c1"],
            "level4_id": ["l1"],
            "difficulty": ["easy"],
            "learning_stage": ["elementary"],
        }
    ).to_csv(content_path, index=False)

    with pytest.raises(AssertionError, match="Log data is empty"):
        load_raw_dataframes(str(log_path), str(user_path), str(content_path))


def test_load_data_for_training_reads_only_requested_month_partitions(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    curated_root = tmp_path / "curated"
    june = curated_root / "year=2024" / "month=06"
    july = curated_root / "year=2024" / "month=07"
    june.mkdir(parents=True)
    july.mkdir(parents=True)
    parquet_paths: list[object] = []
    sql_frames = [
        pd.DataFrame({"uuid": ["u1"], "gender": ["female"]}),
        pd.DataFrame({"ucid": ["c1"]}),
    ]

    def fake_read_parquet(path):
        parquet_paths.append(path)
        return pd.DataFrame(
            {"timestamp_TW": [pd.Timestamp("2024-06-15")], "uuid": ["u1"]}
        )

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(pd, "read_sql", lambda *_args, **_kwargs: sql_frames.pop(0))

    df_log, df_user, df_content = load_data_for_training(
        datetime(2024, 6, 10), datetime(2024, 6, 20), "engine", curated_root
    )

    assert parquet_paths == [june]
    assert list(df_log["uuid"]) == ["u1"]
    assert list(df_user["uuid"]) == ["u1"]
    assert list(df_content["ucid"]) == ["c1"]


def test_load_data_for_training_accepts_utc_aware_dates(monkeypatch, tmp_path):
    curated_root = tmp_path / "curated"
    june = curated_root / "year=2024" / "month=06"
    june.mkdir(parents=True)
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda _path: pd.DataFrame(
            {"timestamp_TW": [pd.Timestamp("2024-06-15")], "uuid": ["u1"]}
        ),
    )
    monkeypatch.setattr(
        pd,
        "read_sql",
        lambda query, *_args, **_kwargs: (
            pd.DataFrame({"uuid": ["u1"]})
            if "user_profile" in query
            else pd.DataFrame({"ucid": ["c1"]})
        ),
    )

    df_log, _, _ = load_data_for_training(
        datetime(2024, 6, 10, tzinfo=timezone.utc),
        datetime(2024, 6, 20, tzinfo=timezone.utc),
        "engine",
        curated_root,
    )

    assert list(df_log["uuid"]) == ["u1"]
