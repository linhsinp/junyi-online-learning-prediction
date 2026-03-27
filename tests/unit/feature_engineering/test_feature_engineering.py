import numpy as np

from junyi_predictor.pipeline.feature_engineering import (
    build_feature_stage,
    create_upid_accuracy_features,
)


def test_create_upid_accuracy_features_only_uses_prior_rows(feature_accuracy_log_df):
    result = create_upid_accuracy_features(feature_accuracy_log_df)

    assert np.isclose(result.loc[0, "v_upid_acc"], 2 / 3)
    assert np.isclose(result.loc[1, "v_upid_acc"], 1.0)
    assert np.isclose(result.loc[2, "v_upid_acc"], 2 / 3)
    assert np.isclose(result.loc[1, "v_uuid_upid_acc"], 1.0)


def test_build_feature_stage_returns_stage_contract_with_matching_row_counts(
    feature_log_df, feature_user_df, feature_content_df
):
    result = build_feature_stage(
        df_log=feature_log_df,
        df_user=feature_user_df,
        df_content=feature_content_df,
    )

    assert result.log.shape[0] == 3
    assert result.concept_proficiency.shape == (3, 2)
    assert result.level4_proficiency.shape == (3, 2)
