from datetime import datetime

from flytekit import workflow

from flyte.tasks.preprocess import (
    preprocess_log_df,
    read_raw_data_into_df,
    save_preprocessed_data,
)

PATH_OUTPUT = "data/demo"


@workflow
def preprocessing_wf() -> str:
    """Flyte workflow to preprocess raw csv files.

    Returns:
        str: Local preprocessed path
    """
    df_log, df_user, df_content = read_raw_data_into_df(
        start_date=datetime(2019, 6, 1), end_date=datetime(2019, 6, 10)
    )
    df_log = preprocess_log_df(df_log, df_user)
    preprocessed_path = save_preprocessed_data(df_log, df_user, df_content, PATH_OUTPUT)
    return preprocessed_path
