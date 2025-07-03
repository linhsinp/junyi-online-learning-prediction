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

    Args:
        start_date (datetime): Start date for data preprocessing.
        end_date (datetime): End date for data preprocessing.

    Returns:
        tuple[StructuredDataset, StructuredDataset, StructuredDataset]`: Preprocessed log, user, and content data.
    """
    df_log, df_user, df_content = load_from_dbt_and_preprocess_data(
        start_date=start_date, end_date=end_date
    )
    return df_log, df_user, df_content
