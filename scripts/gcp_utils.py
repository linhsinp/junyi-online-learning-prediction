""" 
This script provides utility functions to interact with Google Cloud Storage.
It includes functions to download and upload files to a specified bucket.
"""

from google.cloud import storage


def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")


def upload_blob(bucket_name: str, source_file_name: str, destination_blob_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}")
