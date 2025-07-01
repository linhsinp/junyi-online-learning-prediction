from datetime import datetime

import pandas as pd
from flytekit import task
from sqlalchemy import create_engine

from scripts.preprocess import load_df_from_dbt, preprocess_log, save_parquet

# from dotenv import load_dotenv
# load_dotenv()


custom_image = (
    "europe-west3-docker.pkg.dev/junyi-ml-project/junyi-predictor/flyte-gcs-dev:latest"
)


@task(container_image=custom_image)
def read_raw_data_into_df(
    start_date: datetime,
    end_date: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flyte task to read raw data into DataFrames.
    This function loads log data, user information, and content information
    from the database using SQL queries within a specified date range.

    Args:
        start_date (datetime): Start date for filtering log data.
        end_date (datetime): End date for filtering log data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            df_log (pd.DataFrame): The DataFrame containing log data.
            df_user (pd.DataFrame): The DataFrame containing user information.
            df_content (pd.DataFrame): The DataFrame containing content information.
    """
    print(
        f"Reading raw data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}..."
    )

    # db_url = os.environ["DATABASE_URL"]
    db_url = "postgresql://postgres:postgres@localhost:30001/postgres"
    sqlmodel_engine = create_engine(db_url)

    df_log, df_user, df_content = load_df_from_dbt(
        start_date, end_date, sqlmodel_engine
    )

    return df_log, df_user, df_content


@task(container_image=custom_image)
def preprocess_log_df(df_log: pd.DataFrame, df_user: pd.DataFrame) -> pd.DataFrame:
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


@task(container_image=custom_image)
def save_preprocessed_data(
    df_log: pd.DataFrame,
    df_user: pd.DataFrame,
    df_content: pd.DataFrame,
    working_dir: str,
) -> str:
    """Flyte task to save preprocessed data locally to working directory.

    Args:
        df_log (pd.DataFrame): The preprocessed DataFrame containing log data.
        df_user (pd.DataFrame): The preprocessed DataFrame containing user information.
        df_content (pd.DataFrame): The preprocessed DataFrame containing content information.
        working_dir (os.path): Working directory containing data downloaded from GCS.

    Returns:
        os.path: Path to the downloaded data on the local filesystem.
    """
    save_parquet(df_log, df_user, df_content, working_dir)
    print(f"Preprocessed data saved to {working_dir}.")
    return working_dir


if __name__ == "__main__":
    df_log, df_user, df_content = read_raw_data_into_df(
        start_date=datetime(2019, 6, 1), end_date=datetime(2019, 6, 10)
    )
