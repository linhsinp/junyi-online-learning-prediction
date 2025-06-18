from flytekit import workflow

from flyte.tasks.ingest import fetch_from_gcs
from flyte.tasks.preprocess import (
    preprocess_log_df,
    read_raw_data_into_df,
    save_preprocessed_data,
)


@workflow
def preprocessing_wf() -> str:
    """Flyte workflow to preprocess raw csv files.

    Returns:
        str: Local preprocessed path
    """
    raw_path = fetch_from_gcs(prefix="raw")
    df_log, df_user, df_content = read_raw_data_into_df(raw_path)
    df_log = preprocess_log_df(df_log, df_user)
    preprocessed_path = save_preprocessed_data(df_log, df_user, df_content, raw_path)
    return preprocessed_path
