"""
This script provides utility functions to interact with AWS S3 / MinIO Storage.
It includes functions to download and upload files to a specified bucket.
"""

import os

import boto3
from botocore.client import ClientError, Config

BUCKET_NAME = "junyi-ml-data-bucket"
LOCAL_DIR = "data/feature_store"
REMOTE_DIR = "feature_store"
MINIO_ENDPOINT = "http://localhost:30002"
MINIO_BUCKET = "junyi-ml-data-bucket"
ACCESS_KEY = "minio"
SECRET_KEY = "miniostorage"


def upload_folder_to_minio(local_folder: str = LOCAL_DIR, target_dir: str = REMOTE_DIR):
    """Uploads files from a local folder to a MinIO bucket.

    Args:
        local_folder (str): Path to the local folder containing the files to upload.
        target_dir (str): Target prefix (folder path) in the MinIO bucket.
    """
    # Create S3 client for MinIO
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    # Check if bucket exists, otherwise create it
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' already exists.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            s3.create_bucket(Bucket=BUCKET_NAME)
            print(f"Bucket '{BUCKET_NAME}' created.")
        else:
            raise

    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_key = f"{target_dir}/{relative_path}".replace(
                "\\", "/"
            )  # for Windows compatibility

            s3.upload_file(local_path, MINIO_BUCKET, s3_key)
            print(f"Uploaded: {local_path} → s3://{MINIO_BUCKET}/{s3_key}")


# upload_folder_to_minio(LOCAL_DIR, REMOTE_DIR)


def download_data_to_tmp(prefix: str, local_dir: str = "/tmp/data"):
    """
    Downloads data from the specified blob to for process.

    Args:
        prefix (str): The name of the target blob in the bucket.
    """
    # Create S3 client for MinIO
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    os.makedirs(local_dir, exist_ok=True)

    # List all objects under the given prefix
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

    if "Contents" not in response:
        print(f"No objects found in bucket '{BUCKET_NAME}' with prefix '{prefix}'.")
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        relative_path = os.path.relpath(key, prefix)
        local_path = os.path.join(local_dir, relative_path)
        print(local_path)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(BUCKET_NAME, key, local_path)
        print(f"Downloaded: s3://{BUCKET_NAME}/{key} → {local_path}")


# download_data_to_tmp(REMOTE_DIR)
