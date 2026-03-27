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
