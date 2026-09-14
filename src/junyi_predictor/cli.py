"""Small operational commands for local development and container smoke tests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from junyi_predictor.bootstrap.database import (
    reset_database,
    seed_database_from_gcs,
    seed_database_from_raw_files,
)
from junyi_predictor.bootstrap.kaggle import download_kaggle_data
from junyi_predictor.bootstrap.lake import materialize_log_parquet
from junyi_predictor.paths import CONTENT_FILE, CURATED_LOG_DIR, USER_FILE
from junyi_predictor.progress import execution
from junyi_predictor.settings import DataLakeSettings
from junyi_predictor.storage.data_lake import upload_curated_log, upload_dimension_csvs


def main() -> None:
    """Run an explicit local bootstrap command."""
    parser = argparse.ArgumentParser(prog="junyi-predictor")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("download-data")
    subcommands.add_parser("materialize-parquet")
    subcommands.add_parser("reset-db")
    subcommands.add_parser("seed-db")
    subcommands.add_parser("upload-curated-data")
    subcommands.add_parser("upload-dimension-data")
    subcommands.add_parser("seed-db-from-gcs")
    args = parser.parse_args()

    with execution(args.command, service="junyi-bootstrap"):
        if args.command == "download-data":
            download_kaggle_data()
        elif args.command == "materialize-parquet":
            materialize_log_parquet()
        elif args.command == "reset-db":
            reset_database(os.environ["DATABASE_URL"])
        elif args.command == "seed-db":
            seed_database_from_raw_files(os.environ["DATABASE_URL"])
        elif args.command == "seed-db-from-gcs":
            seed_database_from_gcs(
                os.environ["DATA_LAKE_BUCKET"],
                os.getenv("DIMENSION_DATA_PREFIX", "data/dimensions"),
                os.environ["DATABASE_URL"],
            )
        else:
            settings = DataLakeSettings.from_environment()
            if settings.backend != "gcs" or not settings.gcs_bucket:
                raise ValueError(
                    "upload-curated-data requires DATA_LAKE_BACKEND=gcs and DATA_LAKE_BUCKET"
                )
            if args.command == "upload-curated-data":
                upload_curated_log(
                    Path(settings.curated_log_root or CURATED_LOG_DIR),
                    settings.gcs_bucket,
                    settings.gcs_prefix,
                )
            else:
                upload_dimension_csvs(
                    USER_FILE,
                    CONTENT_FILE,
                    settings.gcs_bucket,
                    os.getenv("DIMENSION_DATA_PREFIX", "data/dimensions"),
                )


if __name__ == "__main__":
    main()
