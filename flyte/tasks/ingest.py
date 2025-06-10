from flytekit import task
from scripts.gcs_utils import download_blob

@task
def ingest_data_gcs(bucket: str, blob: str, local_path: str = "/tmp/junyi.csv") -> str:
    download_blob(bucket, blob, local_path)
    return local_path
