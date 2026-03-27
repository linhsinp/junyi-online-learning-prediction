from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from junyi_predictor.pipeline.preprocessing import (
    load_data_from_database,
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


def test_load_data_from_database_queries_log_user_and_content(
    monkeypatch: pytest.MonkeyPatch,
):
    queries: list[tuple[str, object]] = []
    frames = [
        pd.DataFrame({"uuid": ["u1", "u2"]}),
        pd.DataFrame({"uuid": ["u1", "u2"], "gender": ["female", "male"]}),
        pd.DataFrame({"ucid": ["c1"]}),
    ]

    def fake_read_sql(query, engine, params=None):
        queries.append((query.strip(), params))
        return frames.pop(0)

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    df_log, df_user, df_content = load_data_from_database(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        sqlmodel_engine="engine",
    )

    assert list(df_log["uuid"]) == ["u1", "u2"]
    assert list(df_user["uuid"]) == ["u1", "u2"]
    assert list(df_content["ucid"]) == ["c1"]
    assert "FROM log_problem" in queries[0][0]
    assert queries[1][1] == {"selected_uuid": ["u1", "u2"]}
    assert queries[2][0] == "SELECT * FROM info_content;"
