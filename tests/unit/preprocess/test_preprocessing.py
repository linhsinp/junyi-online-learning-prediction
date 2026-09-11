import pandas as pd

from junyi_predictor.pipeline.preprocessing import preprocess_stage


def test_preprocess_stage_merges_user_attributes_and_removes_leakage_columns(
    preprocess_raw_log_df, preprocess_raw_user_df, preprocess_raw_content_df
):
    result = preprocess_stage(
        df_log=preprocess_raw_log_df,
        df_user=preprocess_raw_user_df,
        df_content=preprocess_raw_content_df,
    )

    assert list(result.log["uuid"]) == ["u1", "u2"]
    assert {"female", "male", "unspecified"}.issubset(result.log.columns)
    assert "total_sec_taken" not in result.log.columns
    assert "is_hint_used" not in result.log.columns
    assert result.log.loc[0, "level"] == 3
    assert result.log.loc[1, "level"] == 4
    assert result.log.loc[0, "female"] == 1
    assert result.log.loc[1, "unspecified"] == 1


def test_preprocess_stage_accepts_nullable_boolean_level_flags(
    preprocess_raw_log_df, preprocess_raw_user_df, preprocess_raw_content_df
):
    log = preprocess_raw_log_df.copy()
    log["is_downgrade"] = pd.Series([True, pd.NA], dtype="boolean")
    log["is_upgrade"] = pd.Series([pd.NA, True], dtype="boolean")

    result = preprocess_stage(log, preprocess_raw_user_df, preprocess_raw_content_df)

    assert result.log["level"].tolist() == [3, 4]
