import shutil

from flytekit import workflow

from flyte.tasks.ingest import fetch_from_gcs
from flyte.tasks.train_model import (
    load_features,
    split_data_and_append_matrices,
    train_and_evaluate_model_task,
    transform_data,
)


@workflow
def training_wf() -> str:
    features_path = fetch_from_gcs(prefix="feature_store")
    df_log, m_concept_proficiency, m_proficiency_level4 = load_features(
        features_path=features_path
    )
    X_train, y_train, X_test, y_test = split_data_and_append_matrices(
        df_log, m_concept_proficiency, m_proficiency_level4
    )
    X_train, X_test = transform_data(X_train, X_test)

    train_and_evaluate_model_task(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="DecisionTreeClassifier",
    )
    train_and_evaluate_model_task(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="GradientBoostingClassifier",
    )
    train_and_evaluate_model_task(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="LogisticRegression_L2",
    )
    train_and_evaluate_model_task(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="LogisticRegression_L1",
    )

    # Cleanup after test
    shutil.rmtree("/tmp/data", ignore_errors=True)

    return "Model training and evaluation workflow finished."
