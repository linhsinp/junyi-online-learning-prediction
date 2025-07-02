import os
from datetime import datetime

from flytekit import workflow
from flytekit.types.file import FlyteFile

from flyte.tasks.preprocess import load_from_dbt_and_preprocess_data, save_df_to_s3_task

PATH_OUTPUT = "/tmp/data/preprocessed"
os.makedirs(PATH_OUTPUT, exist_ok=True)


@workflow
def preprocessing_wf() -> tuple[FlyteFile, FlyteFile, FlyteFile]:
    """Flyte workflow to preprocess raw input data from database.

    Returns:
        tuple[FlyteFile, FlyteFile, FlyteFile]: Paths to the preprocessed log, user, and content data files.
    """
    df_log, df_user, df_content = load_from_dbt_and_preprocess_data(
        start_date=datetime(2019, 6, 1), end_date=datetime(2019, 6, 10)
    )
    return save_df_to_s3_task(
        df_log=df_log, df_user=df_user, df_content=df_content, working_dir=PATH_OUTPUT
    )
