import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from flytekit import Resources, task, workflow
from sqlalchemy import create_engine

from scripts.engineer_feature import (
    create_concept_proficiency,
    create_level4_proficiency_matrix,
    create_upid_acc,
)
from scripts.preprocess import load_df_from_dbt, preprocess_log
from scripts.train_model import (
    apply_min_max_transformation,
    split_data_for_train_and_test,
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
    df_log, df_user, df_content = load_df_from_dbt(
        start_date, end_date, sqlmodel_engine
    )
    print(
        "Preprocessing log df by merging with user data, sorting, and encoding categorical variables."
    )
    df_log = preprocess_log(df_log, df_user)

    # Feature engineering
    print("Starting UPID accuracy feature engineering...")
    df_log = create_upid_acc(df_log)
    print("UPID accuracy feature engineering completed.")
    print("Starting concept proficiency feature engineering...")
    list_concept_id = df_content.ucid.to_numpy()
    dict_concept_id = {id: order for order, id in enumerate(list_concept_id)}
    m_concept_proficiency = create_concept_proficiency(
        df_log, list_concept_id, dict_concept_id
    )
    print("Concept proficiency feature engineering completed.")
    print("Starting level-4 proficiency feature engineering...")
    list_concept_id = df_content.ucid.to_numpy()
    dict_concept_id = {id: order for order, id in enumerate(list_concept_id)}
    list_level4_id = df_content.level4_id.unique()
    dict_level4_id = {id: order for order, id in enumerate(list_level4_id)}
    list_user_id = df_user["uuid"].unique()
    dict_user_id = {id: order for order, id in enumerate(list_user_id)}
    m_proficiency_level4 = create_level4_proficiency_matrix(
        df_log,
        df_content,
        list_concept_id,
        dict_concept_id,
        list_level4_id,
        dict_level4_id,
        list_user_id,
        dict_user_id,
    )
    print("Level-4 proficiency feature engineering completed.")

    # Data splitting
    print("Loading features into DataFrames and numpy arrays...")
    X_train, y_train, X_test, y_test = split_data_for_train_and_test(
        df_log,
        m_concept_proficiency,
        m_proficiency_level4,
        num_samples=num_samples,
    )

    # Data transformation
    print("Applying Min-Max transformation to the data...")
    X_train, X_test = apply_min_max_transformation(X_train, X_test)

    # Train and evaluate the model
    metrics = {
        "DecisionTreeClassifier": None,
        "GradientBoostingClassifier": None,
        "LogisticRegression_L2": None,
        "LogisticRegression_L1": None,
    }
    metrics_1 = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, model_type="DecisionTreeClassifier"
    )
    metrics_2 = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, model_type="GradientBoostingClassifier"
    )
    metrics_3 = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, model_type="LogisticRegression_L2"
    )
    metrics_4 = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, model_type="LogisticRegression_L1"
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
