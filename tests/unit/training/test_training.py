import numpy as np

from junyi_predictor.pipeline.training import (
    apply_min_max_transformation,
    split_training_data,
    train_and_evaluate_model,
)


def test_split_training_data_builds_deterministic_stage_contract(
    training_log_df, training_concept_matrix, training_level4_matrix
):
    split = split_training_data(
        df_log=training_log_df,
        m_concept_proficiency=training_concept_matrix,
        m_proficiency_level4=training_level4_matrix,
        num_samples=5,
    )

    assert split.X_train.shape == (4, 11)
    assert split.X_test.shape == (1, 11)
    assert split.y_train.shape == (4,)
    assert split.y_test.shape == (1,)


def test_training_helpers_scale_features_and_train_models(
    simple_binary_training_arrays,
):
    X_train, X_test, y_train, y_test = simple_binary_training_arrays
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
