import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from flytekit import StructuredDataset, task
from flytekit.types.file import FlyteFile
from sqlalchemy import create_engine

from junyi_predictor.pipeline.preprocessing import (
    load_data_from_database,
    preprocess_stage,
)

load_dotenv()


custom_image = "linhsinp/junyi-predictor-image:latest"


@task(container_image=custom_image)
def load_from_dbt_and_preprocess_data(
    start_date: datetime,
    end_date: datetime,
) -> tuple[StructuredDataset, StructuredDataset, StructuredDataset]:
    """
    Flyte task to preprocess DataFrames loaded from database tables.

    This function loads log data, user information, and content information
    from the database using SQL queries within a specified date range,
    preprocesses the log data by merging it with user data, sorting, and encoding categorical variables.

    Args:
        start_date (datetime): Start date for filtering log data.
        end_date (datetime): End date for filtering log data.
        working_dir (str): Working directory to save preprocessed data.
    Returns:
        str: Path to the working directory containing preprocessed data.
    """
    print(
        f"Reading raw data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}..."
    )
    db_url = os.environ["DATABASE_URL"]
    sqlmodel_engine = create_engine(db_url)
    df_log, df_user, df_content = load_data_from_database(
        start_date, end_date, sqlmodel_engine
    )
    stage_output = preprocess_stage(
        df_log=df_log, df_user=df_user, df_content=df_content
    )
    return (
        StructuredDataset(stage_output.log),
        StructuredDataset(stage_output.user),
        StructuredDataset(stage_output.content),
    )


@task(container_image=custom_image)
def save_df_to_s3_task(
    df_log: StructuredDataset,
    df_user: StructuredDataset,
    df_content: StructuredDataset,
    working_dir: str,
) -> tuple[FlyteFile, FlyteFile, FlyteFile]:
    """Flyte task to save DataFrames to S3 as Parquet files.

    This function saves the log data, user information, and content information.

    Args:
        df_log (pd.DataFrame): The DataFrame containing log data.
        df_user (pd.DataFrame): The DataFrame containing user information.
        df_content (pd.DataFrame): The DataFrame containing content information.
        local_path (str): Local path to save the Parquet files.

    Returns:
        tuple[FlyteFile, FlyteFile, FlyteFile]: Paths to the saved Parquet files
        containing log data, user information, and content information.
    """
    # Convert StructuredDataset to actual DataFrames
    df_log = df_log.open(pd.DataFrame).all()
    df_user = df_user.open(pd.DataFrame).all()
    df_content = df_content.open(pd.DataFrame).all()
    print("Saving preprocessed data to S3 as Parquet files...")
    df_log.to_parquet(os.path.join(working_dir, "Processed_Log_Problem.parquet.gzip"))
    log_path = f"{working_dir}/Processed_Log_Problem.parquet.gzip"
    df_user.to_parquet(
        os.path.join(working_dir, "Processed_Info_UserData.parquet.gzip")
    )
    user_path = f"{working_dir}/Processed_Info_UserData.parquet.gzip"
    df_content.to_parquet(
        os.path.join(working_dir, "Processed_Info_Content.parquet.gzip")
    )
    content_path = f"{working_dir}/Processed_Info_Content.parquet.gzip"
    # Return as FlyteFile, which will handle upload to S3/Minio
    return FlyteFile(log_path), FlyteFile(user_path), FlyteFile(content_path)
