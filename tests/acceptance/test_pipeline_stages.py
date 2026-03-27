import pandas as pd

from junyi_predictor.pipeline.feature_engineering import build_feature_stage
from junyi_predictor.pipeline.preprocessing import preprocess_stage
from junyi_predictor.pipeline.training import split_training_data


def test_preprocess_feature_and_training_stages_share_explicit_contracts():
    df_log = pd.DataFrame(
        {
            "timestamp_TW": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "uuid": ["u1", "u1", "u2", "u2", "u3"],
            "ucid": ["c1", "c2", "c1", "c2", "c1"],
            "upid": ["p1", "p2", "p1", "p2", "p1"],
            "problem_number": [1, 2, 1, 2, 3],
            "exercise_problem_repeat_session": [0, 1, 0, 1, 0],
            "is_correct": [True, False, True, False, True],
            "total_sec_taken": [11, 12, 13, 14, 15],
            "is_hint_used": [False, True, False, True, False],
            "is_downgrade": [0, 0, 1, 0, 0],
            "is_upgrade": [0, 1, 0, 0, 0],
            "level": [1, 2, 3, 4, 5],
        }
    )
    df_user = pd.DataFrame(
        {
            "uuid": ["u1", "u2", "u3"],
            "user_grade": [5, 6, 7],
            "gender": ["female", "male", None],
        }
    )
    df_content = pd.DataFrame({"ucid": ["c1", "c2"], "level4_id": ["l1", "l2"]})

    preprocessed = preprocess_stage(
        df_log=df_log, df_user=df_user, df_content=df_content
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
