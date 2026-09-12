"""Google Cloud Storage helpers used by the pipeline."""

import logging
import os

from google.cloud import storage

from junyi_predictor.paths import FEATURE_STORE_DIR
from junyi_predictor.progress import timed

logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv("GCS_BUCKET", "junyi-ml-data-bucket")
LOCAL_DIR = str(FEATURE_STORE_DIR)
REMOTE_DIR = "feature_store"


@timed("gcs.download", fields=("source_blob_name",))
def download_blob(
    bucket_name: str, source_blob_name: str, destination_file_name: str
) -> None:
    """Download a blob from Google Cloud Storage to a local file."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(
        "Blob downloaded", extra={"event": "gcs.downloaded", "key": source_blob_name}
    )


@timed("gcs.upload", fields=("destination_blob_name",))
def upload_blob(
    bucket_name: str, source_file_name: str, destination_blob_name: str
) -> None:
    """Upload a local file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logger.info(
        "Blob uploaded", extra={"event": "gcs.uploaded", "key": destination_blob_name}
    )


@timed("gcs.upload_folder", fields=("target_dir",))
def upload_folder(local_folder: str = LOCAL_DIR, target_dir: str = REMOTE_DIR) -> None:
    """Upload every file from a local folder to a target directory in GCS."""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            blob_path = os.path.relpath(local_path, local_folder)
            blob = bucket.blob(f"{target_dir}/{blob_path}")
            blob.upload_from_filename(local_path)
            logger.info(
                "Blob uploaded",
                extra={"event": "gcs.uploaded", "key": f"{target_dir}/{blob_path}"},
            )


@timed("gcs.download_prefix", fields=("prefix",))
def download_data_to_tmp(prefix: str, local_dir: str = "/tmp/data") -> None:
    """Download all blobs matching a prefix into a local directory."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    os.makedirs(local_dir, exist_ok=True)

    for blob in client.list_blobs(bucket, prefix=prefix):
        relative_path = blob.name[len(prefix) :].lstrip("/")
        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info(
            "Blob downloaded", extra={"event": "gcs.downloaded", "key": blob.name}
        )
