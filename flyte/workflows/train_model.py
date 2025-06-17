import numpy as np
from flytekit import dynamic, workflow
from tasks.ingest import fetch_from_gcs
from tasks.train_model import (
    ModelType,
    load_features,
    split_data_and_append_matrices,
    train_and_evaluate_model_task,
    transform_data,
)


@dynamic
def evaluate_multiple_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_path: str,
    models_to_evaluate: list[ModelType],
) -> dict:
    results = {}

    for model_type in models_to_evaluate:
        metrics = train_and_evaluate_model_task(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_type=model_type,
            model_path=f"{model_path}_{model_type.value}.pkl",
        )
        results[model_type.value] = metrics

    return results


@workflow
def training_wf(
    models_to_evaluate: list[ModelType] = [
        ModelType.DecisionTree,
        ModelType.GradientBoosting,
    ],
    model_path="tmp/data/model",
) -> dict:
    features_path = fetch_from_gcs(prefix="feature_store")
    df_log, m_concept_proficiency, m_proficiency_level4 = load_features(
        features_path=features_path
    )
    X_train, y_train, X_test, y_test = split_data_and_append_matrices(
        df_log, m_concept_proficiency, m_proficiency_level4
    )
    X_train, X_test = transform_data(X_train, X_test)
    return evaluate_multiple_models(
        X_train, y_train, X_test, y_test, model_path, models_to_evaluate
    )
