import numpy as np
import pandas as pd

from junyi_predictor.pipeline.training import (
    apply_min_max_transformation,
    split_training_data,
    train_and_evaluate_model,
)


def test_split_training_data_builds_deterministic_stage_contract():
    df_log = pd.DataFrame(
        {
            "user_grade": [5, 6, 5, 6, 7],
            "female": [1, 0, 1, 0, 0],
            "male": [0, 1, 0, 1, 0],
            "unspecified": [0, 0, 0, 0, 1],
            "v_upid_acc": [0.2, 0.4, 0.6, 0.8, 1.0],
            "level": [1, 2, 3, 4, 5],
            "problem_number": [1, 2, 3, 4, 5],
            "exercise_problem_repeat_session": [0, 1, 0, 1, 0],
            "is_correct": [True, False, True, False, True],
        }
    )
    concept = np.arange(10, dtype=float).reshape(5, 2)
    level4 = np.arange(5, dtype=float).reshape(5, 1)

    split = split_training_data(
        df_log=df_log,
        m_concept_proficiency=concept,
        m_proficiency_level4=level4,
        num_samples=5,
    )

    assert split.X_train.shape == (4, 11)
    assert split.X_test.shape == (1, 11)
    assert split.y_train.shape == (4,)
    assert split.y_test.shape == (1,)


def test_training_helpers_scale_features_and_train_models():
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    X_test = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_train = np.array([0, 1, 0, 1], dtype=bool)
    y_test = np.array([0, 1], dtype=bool)

    X_train_scaled, X_test_scaled = apply_min_max_transformation(X_train, X_test)
    metrics = train_and_evaluate_model(
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        model_type="DecisionTreeClassifier",
    )

    assert np.all(X_train_scaled >= 0)
    assert np.all(X_train_scaled <= 1)
    assert metrics["train_score"] >= 0.5
    assert metrics["test_score"] >= 0.5
