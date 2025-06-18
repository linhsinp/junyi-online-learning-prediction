import os

import pandas as pd
from flytekit import task

from scripts.preprocess import load_data_into_df, preprocess_log, save_parquet


@task
def read_raw_data_into_df(working_dir: os.path):
    """Flyte task to read raw csv files into pandas dataframe.

    Args:
        working_dir (os.path): Working directory containing data downloaded from GCS.

    Returns:
        df_log (pd.DataFrame): The DataFrame containing log data.
        df_user (pd.DataFrame): The DataFrame containing user information.
        df_content (pd.DataFrame): The DataFrame containing content information.
    """
    path_log_full = os.path.join(working_dir, "Log_Problem.csv")
    path_user = os.path.join(working_dir, "Info_UserData.csv")
    path_content = os.path.join(working_dir, "Info_Content.csv")
    df_log, df_user, df_content = load_data_into_df(
        path_log_full, path_user, path_content
    )
    return df_log, df_user, df_content


@task
def preprocess_log_df(df_log: pd.DataFrame, df_user: pd.DataFrame):
    """
    Flyte task to preprocess the log DataFrame.

    Args:
        df_log (pd.DataFrame): The DataFrame containing log data.
        df_user (pd.DataFrame): The DataFrame containing user information.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    print(
        "Preprocessing log df by merging with user data, sorting, and encoding categorical variables."
    )
    df_log = preprocess_log(df_log, df_user)
    return df_log


@task
def save_preprocessed_data(
    df_log: pd.DataFrame,
    df_user: pd.DataFrame,
    df_content: pd.DataFrame,
    working_dir: os.path,
) -> os.path:
    """Flyte task to save preprocessed data locally to working directory.

    Args:
        df_log (pd.DataFrame): The preprocessed DataFrame containing log data.
        df_user (pd.DataFrame): The preprocessedDataFrame containing user information.
        df_content (pd.DataFrame): The preprocessedDataFrame containing content information.
        working_dir (os.path): Working directory containing data downloaded from GCS.

    Returns:
        os.path: Path to the downloaded data on the local filesystem.
    """
    save_parquet(df_log, df_user, df_content, working_dir)
    print(f"Preprocessed data saved to {working_dir}.")
    return working_dir
