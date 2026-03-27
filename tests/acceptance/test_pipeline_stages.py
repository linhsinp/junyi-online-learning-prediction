from junyi_predictor.pipeline.feature_engineering import build_feature_stage
from junyi_predictor.pipeline.preprocessing import preprocess_stage
from junyi_predictor.pipeline.training import split_training_data


def test_preprocess_feature_and_training_stages_share_explicit_contracts(
    pipeline_raw_log_df, pipeline_raw_user_df, pipeline_raw_content_df
):
    preprocessed = preprocess_stage(
        df_log=pipeline_raw_log_df,
        df_user=pipeline_raw_user_df,
        df_content=pipeline_raw_content_df,
    )
    featured = build_feature_stage(
        df_log=preprocessed.log,
        df_user=preprocessed.user,
        df_content=preprocessed.content,
    )
    split = split_training_data(
        df_log=featured.log,
        m_concept_proficiency=featured.concept_proficiency,
        m_proficiency_level4=featured.level4_proficiency,
    )

    assert featured.log.shape[0] == preprocessed.log.shape[0]
    assert featured.concept_proficiency.shape[0] == preprocessed.log.shape[0]
    assert featured.level4_proficiency.shape[0] == preprocessed.log.shape[0]
    assert split.X_train.shape[0] + split.X_test.shape[0] == preprocessed.log.shape[0]
