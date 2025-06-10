"""
This script downloads a dataset from Kaggle using the Kaggle API.
Make sure to have the Kaggle API installed and configured with your credentials.
"""

import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_data(output_dir="data/raw"):
    """Download the Junyi Academy Learning Activity Public Dataset from Kaggle.

    * Info_Content
    * Info_UserData
    * Log_Problem
    
    Args:
        output_dir (str): Directory where the dataset will be downloaded and extracted.

    """
    dataset = "junyiacademy/learning-activity-public-dataset-by-junyi-academy"
    os.makedirs(output_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=output_dir, unzip=True)


if __name__ == "__main__":
    download_kaggle_data()
