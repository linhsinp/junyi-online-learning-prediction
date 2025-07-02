import os
from datetime import datetime

from flytekit import workflow
from flytekit.types.structured import StructuredDataset

from flyte.tasks.preprocess import load_from_dbt_and_preprocess_data

PATH_OUTPUT = "/tmp/data/preprocessed"
os.makedirs(PATH_OUTPUT, exist_ok=True)


@workflow
def preprocessing_wf(
    start_date: datetime = datetime(2019, 6, 1),
    end_date: datetime = datetime(2019, 6, 10),
) -> tuple[StructuredDataset, StructuredDataset, StructuredDataset]:
    """Flyte workflow to preprocess raw input data from database.

    Returns:
        tuple[FlyteFile, FlyteFile, FlyteFile]: Paths to the preprocessed log, user, and content data files.
    """
    df_log, df_user, df_content = load_from_dbt_and_preprocess_data(
        start_date=start_date, end_date=end_date
    )
    # If your task returns DataFrames, just return them directly
    return df_log, df_user, df_content
