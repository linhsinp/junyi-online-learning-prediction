import os
from typing import Literal

from flytekit import task

from scripts.gcs_utils import download_data_to_tmp


@task
def fetch_from_gcs(
    prefix: Literal["raw", "experiment", "feature_store", "model"],
    local_dir: str = "/tmp/data",
) -> str:
    """
    General Flyte task to download data from GCS to a specified local directory.

    Args:
        prefix (str): The folder in the GCS bucket to download from.
        local_dir (str): Local root directory to download data into.

    Returns:
        str: Path to the downloaded data on the local filesystem.
    """
    print(f"Downloading GCS data from prefix '{prefix}' to local dir '{local_dir}'")
    download_data_to_tmp(prefix=prefix, local_dir=local_dir)
    return os.path.join(local_dir, prefix)
