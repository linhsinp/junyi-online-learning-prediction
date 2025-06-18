import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from flytekit import task

from scripts.train_model import (
    apply_min_max_transformation,
    load_data_into_df,
    load_matrices,
    split_data_for_train_and_test,
    train_and_evaluate_model,
)


@task
def load_features(
    features_path: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Flyte task to load features into dfs and np.arrays.

    Args:
        features_path (FlyteDirectory: Working directory containing data downloaded from GCS.

    Returns:
        Dataframes with features and additional matrices.
    """
    print("Loading features into dfs and numpy arrays...")
    path_log_full = os.path.join(features_path, "df_log_with_upid_acc.parquet.gzip")
    df_log = load_data_into_df(path_log_full)
    path_m_concept_proficiency = os.path.join(
        features_path, "m_concept_proficiency.npz"
    )
    path_m_proficiency_level4 = os.path.join(features_path, "m_proficiency_level4.npz")
    m_concept_proficiency, m_proficiency_level4 = load_matrices(
        path_m_concept_proficiency, path_m_proficiency_level4
    )
    return df_log, m_concept_proficiency, m_proficiency_level4


@task
def split_data_and_append_matrices(
    df_log: pd.DataFrame,
    m_concept_proficiency: np.ndarray,
    m_proficiency_level4: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flyte task to split data into 80-20% train-test, and append additional matrices.

    Args:
        df_log (pd.DataFrame): The DataFrame containing log data.

    Returns:
        Train and test sets in np.arrays
    """
    print(
        "Splitting split data into 80-20% train-test, and append additional matrices..."
    )
    X_train, y_train, X_test, y_test = split_data_for_train_and_test(
        df_log, m_concept_proficiency, m_proficiency_level4
    )
    return X_train, y_train, X_test, y_test


@task
def transform_data(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Flyte task to perform min-max transformation.

    Args:
        X_train (np.ndarray): Feature set for training.
        X_test (np.ndarray): Feature set for testing.

    Returns:
        Train and test sets in np.arrays
    """
    print("Performing min-max transformation...")
    X_train, X_test = apply_min_max_transformation(X_train, X_test)
    return X_train, X_test


@task
def train_and_evaluate_model_task(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    model_path: str = "/tmp/model",
) -> Dict:
    """Flyte task to train and evaluate models.

    Args:
        X_train (np.ndarray): Feature set for training.
        y_train (np.ndarray): Predicted variable for training.
        X_test (np.ndarray): Feature set for testing.
        y_test (np.ndarray): Predicted variable for testing.
        model_type (class): A class of model name pairs for enumeration.
        model_path (FlyteDirectory):

    Returns:
        Evaluation metrics of train and test scores (accuracy).
    """
    return train_and_evaluate_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_type=model_type,
        model_path=model_path,
    )
