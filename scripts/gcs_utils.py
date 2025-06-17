""" 
This script provides utility functions to interact with Google Cloud Storage.
It includes functions to download and upload files to a specified bucket.
"""

from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("GCS_BUCKET") 
LOCAL_DIR = "data/feature_store"
REMOTE_DIR = "feature_store"


def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """
    Downloads a blob from the specified bucket to a local file.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_blob_name (str): The name of the blob in the bucket.
        destination_file_name (str): The local file path where the blob will be downloaded.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")


def upload_blob(bucket_name: str, source_file_name: str, destination_blob_name: str):
    """
    Uploads a file to the specified bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_file_name (str): The local file path to upload.
        destination_blob_name (str): The name of the blob in the bucket.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}")


def upload_folder(local_folder: str = LOCAL_DIR, target_dir: str = REMOTE_DIR):
    """Uploads files from a local folder to Google Cloud Storage.
    
    Useful for batching uploads of multiple files to a specific directory in the GCS bucket.

    Args:
        local_folder (str): Path to the local folder containing the files to upload.
        REMOTE_dir (str): Target directory in the GCS bucket where files will be uploaded.
    """
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            blob_path = os.path.relpath(local_path, local_folder)
            blob = bucket.blob(f"{target_dir}/{blob_path}")
            blob.upload_from_filename(local_path)
            print(f"Uploaded: {local_path} → gs://{BUCKET_NAME}/{target_dir}/{blob_path}")


def download_data_to_tmp(prefix: str, local_dir: str = "/tmp/data"):
    """
    Downloads data from the specified blob to for process.

    Args:
        prefix (str): The name of the target blob in the bucket.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    os.makedirs(local_dir, exist_ok=True)

    for blob in client.list_blobs(bucket, prefix=prefix):
        relative_path = blob.name[len(prefix):].lstrip("/")
        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded: gs://{BUCKET_NAME}/{blob.name} → {local_path}")
