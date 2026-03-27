import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from flytekit import Resources, task, workflow
from sqlalchemy import create_engine

from junyi_predictor.pipeline.feature_engineering import (
    build_feature_stage,
)
from junyi_predictor.pipeline.preprocessing import (
    load_data_from_database,
    preprocess_stage,
)
from junyi_predictor.pipeline.training import (
    apply_min_max_transformation,
    split_training_data,
    train_and_evaluate_model,
)

load_dotenv()


custom_image = "linhsinp/junyi-predictor-image:latest"


@task(
    container_image=custom_image,
    limits=Resources(mem="4Gi"),
    requests=Resources(mem="2Gi"),
)
def full_pipeline_task(
    start_date: datetime = datetime(2019, 6, 1),
    end_date: datetime = datetime(2019, 6, 10),
    num_samples: Optional[
        int
    ] = None,  # Number of samples to use for training and testing
) -> dict:
    """
    Full ML pipeline: raw ingestion → preprocessing → feature engineering → training and evaluation.

    This function orchestrates the entire process, from loading raw data to training and evaluating models.

    Args:
        start_date (datetime): Start date for data ingestion.
        end_date (datetime): End date for data ingestion.
        num_samples (int): Number of samples to use for training and testing. If None,
                           all available data will be used.

    Returns:
        dict: A dictionary containing evaluation metrics for each model.
    """
    # Load raw data from the database and preprocess it
    print(
        f"Reading raw data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}..."
    )
    db_url = os.environ["DATABASE_URL"]
    sqlmodel_engine = create_engine(db_url)
    df_log, df_user, df_content = load_data_from_database(
        start_date, end_date, sqlmodel_engine
    )
    preprocessed = preprocess_stage(
        df_log=df_log, df_user=df_user, df_content=df_content
    )

    feature_output = build_feature_stage(
        df_log=preprocessed.log,
        df_user=preprocessed.user,
        df_content=preprocessed.content,
    )

    # Data splitting
    print("Loading features into DataFrames and numpy arrays...")
    split = split_training_data(
        feature_output.log,
        feature_output.concept_proficiency,
        feature_output.level4_proficiency,
        num_samples=num_samples,
    )

    # Data transformation
    print("Applying Min-Max transformation to the data...")
    X_train, X_test = apply_min_max_transformation(split.X_train, split.X_test)

    # Train and evaluate the model
    metrics = {
        "DecisionTreeClassifier": None,
        "GradientBoostingClassifier": None,
        "LogisticRegression_L2": None,
        "LogisticRegression_L1": None,
    }
    metrics_1 = train_and_evaluate_model(
        X_train,
        split.y_train,
        X_test,
        split.y_test,
        model_type="DecisionTreeClassifier",
    )
    metrics_2 = train_and_evaluate_model(
        X_train,
        split.y_train,
        X_test,
        split.y_test,
        model_type="GradientBoostingClassifier",
    )
    metrics_3 = train_and_evaluate_model(
        X_train, split.y_train, X_test, split.y_test, model_type="LogisticRegression_L2"
    )
    metrics_4 = train_and_evaluate_model(
        X_train, split.y_train, X_test, split.y_test, model_type="LogisticRegression_L1"
    )
    metrics["DecisionTreeClassifier"] = metrics_1
    metrics["GradientBoostingClassifier"] = metrics_2
    metrics["LogisticRegression_L2"] = metrics_3
    metrics["LogisticRegression_L1"] = metrics_4

    return metrics


@workflow
def full_pipeline_wf(
    start_date: datetime = datetime(2019, 6, 1),
    end_date: datetime = datetime(2019, 6, 10),
    num_samples: Optional[
        int
    ] = 50000,  # Number of samples to use for training and testing
) -> dict:
    """
    Full ML pipeline workflow that orchestrates the entire process from raw data ingestion to model training and evaluation.

    Args:
        start_date (datetime): Start date for data ingestion.
        end_date (datetime): End date for data ingestion.
        num_samples (int): Number of samples to use for training and testing. If None,
                           all available data will be used.

    Returns:
        dict: A dictionary containing evaluation metrics for each model.
    """
    return full_pipeline_task(
        start_date=start_date, end_date=end_date, num_samples=num_samples
    )
