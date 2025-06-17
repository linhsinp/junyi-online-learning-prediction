"""
PROBLEM-LEVEL PREDICTION PREPROCESSING
The aim is to predict whether a student will answer a problem correct given the details of the problem and the student's performance history.
"""

import os

import pandas as pd

# For training, exclude the selected variables which do not exist before a student takes the exercise
VARS_REDUNDANT = ["total_sec_taken", "is_hint_used", "is_downgrade", "is_upgrade"]
VARS_LOG_CATEGORY = ["uuid", "ucid", "upid"]
VARS_CONTENT_CATEGORY = ["level3_id", "level4_id"]

# Path
PATH_INPUT = "data/raw"
PATH_OUTPUT = "data/output"
PATH_EXPERIMENT = "data/experiment"
PATH_TEST = "data/test"

# Files
FILE_LOG_FULL = os.path.join(PATH_INPUT, "Log_Problem.csv")
FILE_USER = os.path.join(PATH_INPUT, "Info_UserData.csv")
FILE_CONTENT = os.path.join(PATH_INPUT, "Info_Content.csv")


def load_data_into_df(
    path_log_full: str, path_user: str, path_content: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw data into pandas dataframe.

    Args:
        path_log_full (str): Path to raw df_log file
        path_user (str): Path to raw df_user file
        path_content (str): Path to raw df_content file

    Returns:
        df_log (pd.DataFrame): The DataFrame containing log data.
        df_user (pd.DataFrame): The DataFrame containing user information.
        df_content (pd.DataFrame): The DataFrame containing content information.
    """
    log_dtypes = {
        "timestamp_TW": "object",
        "uuid": "category",
        "ucid": "category",
        "upid": "category",
        "problem_number": "int16",
        "exercise_problem_repeat_session": "int16",
        "is_correct": "boolean",
        "total_sec_taken": "int16",
        "total_attempt_cnt": "int16",
        "used_hint_cnt": "int16",
        "is_hint_used": "boolean",
        # 'is_downgrade':'boolean',
        # 'is_upgrade':'boolean',
        "level": "int8",
    }
    user_dtype = {"uuid": "category", "gender": "category", "user_grade": "int8"}
    content_dtype = {
        "ucid": "category",
        "level4_id": "category",
        "difficulty": "category",
        "learning_stage": "category",
    }

    df_log = pd.read_csv(path_log_full, dtype=log_dtypes)
    df_user = pd.read_csv(path_user, dtype=user_dtype)
    df_content = pd.read_csv(path_content, dtype=content_dtype)

    assert not df_log.empty, "Log data is empty."
    assert not df_user.empty, "User data is empty."
    assert not df_content.empty, "Content data is empty."

    return df_log, df_user, df_content


def preprocess_log(df_log: pd.DataFrame, df_user: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the log DataFrame by merging with user data, sorting, and encoding categorical variables.

    Args:
        df_log (pd.DataFrame): The DataFrame containing log data.
        df_user (pd.DataFrame): The DataFrame containing user information.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df_log = pd.merge(
        df_log, df_user[["uuid", "user_grade", "gender"]], on="uuid", how="left"
    )
    df_log = df_log.sort_values(["timestamp_TW", "uuid", "upid"]).reset_index(drop=True)

    # Convert gender to one-hot encoding to handle "unspecified"; set NaN as "unspecified"
    df_log.fillna(value={"gender": "unspecified"}, inplace=True)
    df_log = pd.concat([df_log, pd.get_dummies(df_log.gender)], axis=1).drop(
        columns="gender"
    )

    # Redefine "level" as the "uuid" level of this exercise BEFORE the attempt; offsetting the change due to this attempt
    df_log["level"] = (
        df_log["level"]
        + df_log["is_downgrade"].fillna(0).astype(int)
        - df_log["is_upgrade"].fillna(0).astype(int)
    ).astype("int8")

    df_log = df_log.drop(columns=VARS_REDUNDANT)

    return df_log


def save_parquet(
    df_log: pd.DataFrame,
    df_user: pd.DataFrame,
    df_content: pd.DataFrame,
    path_output: str,
) -> None:
    """
    Save the preprocessed DataFrames to Parquet files.

    Args:
        df_log (pd.DataFrame): The preprocessed log DataFrame.
        df_user (pd.DataFrame): The user information DataFrame.
        df_content (pd.DataFrame): The content information DataFrame.
    """
    df_log.to_parquet(
        os.path.join(path_output, "Processed_Log_Problem_raw_timestamp.parquet.gzip")
    )
    df_user.to_parquet(
        os.path.join(path_output, "Processed_Info_UserData.parquet.gzip")
    )
    df_content.to_parquet(
        os.path.join(path_output, "Processed_Info_Content.parquet.gzip")
    )


def split_experiment_data(
    df_log: pd.DataFrame, df_user: pd.DataFrame, df_content: pd.DataFrame
) -> None:
    """
    Split the data into training and testing sets for benchmarking purposes.

    This function trains the benchmark on the first 10,000 users and tests it on the rest.
    It saves the training and testing sets as Parquet files.

    Args:
        df_log (pd.DataFrame): The preprocessed log DataFrame.
        df_user (pd.DataFrame): The user information DataFrame.
        df_content (pd.DataFrame): The content information DataFrame.
    """
    # Experiment on first 10,000 users
    df_user_train = df_user[
        df_user["uuid"].isin(df_log["uuid"].unique()[:10000])
    ].reset_index(drop=True)
    df_user_train["uuid"] = df_user_train["uuid"].cat.remove_unused_categories()
    df_log_train = df_log[df_log["uuid"].isin(df_user_train["uuid"])].reset_index(
        drop=True
    )
    df_log_train["uuid"] = df_log_train["uuid"].cat.remove_unused_categories()
    df_content_train = df_content[
        df_content["ucid"].isin(df_log_train["ucid"].unique())
    ].reset_index(drop=True)
    df_log_train.to_parquet(
        os.path.join(PATH_EXPERIMENT, "Processed_Log_Problem_train.parquet.gzip")
    )
    df_user_train.to_parquet(
        os.path.join(PATH_EXPERIMENT, "Processed_Info_UserData_train.parquet.gzip")
    )
    df_content_train.to_parquet(
        os.path.join(PATH_EXPERIMENT, "Processed_Info_Content_train.parquet.gzip")
    )
    # Keep the rest for further training
    df_user_test = df_user[~df_user["uuid"].isin(df_user_train["uuid"])].reset_index(
        drop=True
    )
    df_user_test["uuid"] = df_user_test["uuid"].cat.remove_unused_categories()
    df_log_test = df_log[~df_log["uuid"].isin(df_user_train["uuid"])].reset_index(
        drop=True
    )
    df_log_test["uuid"] = df_log_test["uuid"].cat.remove_unused_categories()
    df_content_test = df_content[
        df_content["ucid"].isin(df_log_test["ucid"].unique())
    ].reset_index(drop=True)
    df_log_test.to_parquet(
        os.path.join(PATH_TEST, "Processed_Log_Problem_test.parquet.gzip")
    )
    df_user_test.to_parquet(
        os.path.join(PATH_TEST, "Processed_Info_UserData_test.parquet.gzip")
    )
    df_content_test.to_parquet(
        os.path.join(PATH_TEST, "Processed_Info_Content_test.parquet.gzip")
    )


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    os.makedirs(PATH_EXPERIMENT, exist_ok=True)
    os.makedirs(PATH_TEST, exist_ok=True)

    # Load raw data
    df_log, df_user, df_content = load_data_into_df(
        FILE_LOG_FULL, FILE_USER, FILE_CONTENT
    )

    # Preprocess log data
    df_log = preprocess_log(df_log, df_user)

    # Save the preprocessed DataFrames
    save_parquet(df_log, df_user, df_content)

    # Split the data into training and testing sets
    split_experiment_data(df_log, df_user, df_content)
    print("Preprocessing completed and data saved successfully.")
