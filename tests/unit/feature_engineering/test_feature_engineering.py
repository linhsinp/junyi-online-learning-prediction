import numpy as np
import pandas as pd

from junyi_predictor.pipeline.feature_engineering import (
    build_feature_stage,
    create_upid_accuracy_features,
)


def test_create_upid_accuracy_features_only_uses_prior_rows():
    df_log = pd.DataFrame(
        {
            "uuid": ["u1", "u1", "u2"],
            "ucid": ["c1", "c1", "c2"],
            "upid": ["p1", "p1", "p2"],
            "level": [1, 2, 3],
            "is_correct": [True, False, True],
            "user_grade": [5, 5, 6],
            "female": [1, 1, 0],
            "male": [0, 0, 0],
            "unspecified": [0, 0, 1],
            "problem_number": [1, 2, 1],
            "exercise_problem_repeat_session": [0, 1, 0],
        }
    )

    result = create_upid_accuracy_features(df_log)

    assert np.isclose(result.loc[0, "v_upid_acc"], 2 / 3)
    assert np.isclose(result.loc[1, "v_upid_acc"], 1.0)
    assert np.isclose(result.loc[2, "v_upid_acc"], 2 / 3)
    assert np.isclose(result.loc[1, "v_uuid_upid_acc"], 1.0)


def test_build_feature_stage_returns_stage_contract_with_matching_row_counts():
    df_log = pd.DataFrame(
        {
            "uuid": ["u1", "u1", "u2"],
            "ucid": ["c1", "c2", "c1"],
            "upid": ["p1", "p2", "p1"],
            "level": [1, 2, 3],
            "is_correct": [True, False, True],
            "user_grade": [5, 5, 6],
            "female": [1, 1, 0],
            "male": [0, 0, 1],
            "unspecified": [0, 0, 0],
            "problem_number": [1, 2, 1],
            "exercise_problem_repeat_session": [0, 1, 0],
        }
    )
    df_user = pd.DataFrame({"uuid": ["u1", "u2"]})
    df_content = pd.DataFrame({"ucid": ["c1", "c2"], "level4_id": ["l1", "l2"]})

    result = build_feature_stage(df_log=df_log, df_user=df_user, df_content=df_content)

    assert result.log.shape[0] == 3
    assert result.concept_proficiency.shape == (3, 2)
    assert result.level4_proficiency.shape == (3, 2)
