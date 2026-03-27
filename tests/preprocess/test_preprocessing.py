import pandas as pd

from junyi_predictor.pipeline.preprocessing import preprocess_stage


def test_preprocess_stage_merges_user_attributes_and_removes_leakage_columns():
    df_log = pd.DataFrame(
        {
            "timestamp_TW": ["2024-01-02", "2024-01-01"],
            "uuid": ["u2", "u1"],
            "ucid": ["c2", "c1"],
            "upid": ["p2", "p1"],
            "problem_number": [2, 1],
            "exercise_problem_repeat_session": [0, 1],
            "is_correct": [False, True],
            "total_sec_taken": [20, 10],
            "is_hint_used": [False, True],
            "is_downgrade": [1, 0],
            "is_upgrade": [0, 1],
            "level": [3, 4],
        }
    )
    df_user = pd.DataFrame(
        {
            "uuid": ["u1", "u2"],
            "user_grade": [5, 6],
            "gender": ["female", None],
        }
    )
    df_content = pd.DataFrame({"ucid": ["c1", "c2"], "level4_id": ["l1", "l2"]})

    result = preprocess_stage(df_log=df_log, df_user=df_user, df_content=df_content)

    assert list(result.log["uuid"]) == ["u1", "u2"]
    assert {"female", "male", "unspecified"}.issubset(result.log.columns)
    assert "total_sec_taken" not in result.log.columns
    assert "is_hint_used" not in result.log.columns
    assert result.log.loc[0, "level"] == 3
    assert result.log.loc[1, "level"] == 4
    assert result.log.loc[0, "female"] == 1
    assert result.log.loc[1, "unspecified"] == 1
