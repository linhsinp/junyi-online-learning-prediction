"""Kaggle bootstrap helpers."""

from __future__ import annotations

from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from junyi_predictor.paths import RAW_DATA_DIR

KAGGLE_DATASET = "junyiacademy/learning-activity-public-dataset-by-junyi-academy"


def download_kaggle_data(output_dir: str | Path = RAW_DATA_DIR) -> None:
    """Download and extract the Junyi raw CSV dataset into the raw artifact path."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(destination), unzip=True)
