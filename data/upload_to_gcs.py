"""Upload data to Google Cloud Storage."""

import os
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("GCS_BUCKET") 
LOCAL_DIR = "data/raw"
TARGET_DIR = "raw"


def upload_to_gcs(local_folder=LOCAL_DIR, target_dir=TARGET_DIR):
    """Uploads Junyi raw files from a local folder to Google Cloud Storage.
    
    * Info_Content
    * Info_UserData
    * Log_Problem: Note that this file is 3 GB, so it may take a while to upload.

    Args:
        local_folder (str): Path to the local folder containing the files to upload.
    
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


if __name__ == "__main__":
    upload_to_gcs()