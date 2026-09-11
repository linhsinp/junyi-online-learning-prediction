"""Small operational commands for local development and container smoke tests."""

from __future__ import annotations

import argparse
import os

from junyi_predictor.bootstrap.database import (
    reset_database,
    seed_database_from_raw_files,
)
from junyi_predictor.bootstrap.kaggle import download_kaggle_data
from junyi_predictor.bootstrap.lake import materialize_log_parquet


def main() -> None:
    """Run an explicit local bootstrap command."""
    parser = argparse.ArgumentParser(prog="junyi-predictor")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("download-data")
    subcommands.add_parser("materialize-parquet")
    subcommands.add_parser("reset-db")
    subcommands.add_parser("seed-db")
    args = parser.parse_args()

    if args.command == "download-data":
        download_kaggle_data()
    elif args.command == "materialize-parquet":
        materialize_log_parquet()
    elif args.command == "reset-db":
        reset_database(os.environ["DATABASE_URL"])
    else:
        seed_database_from_raw_files(os.environ["DATABASE_URL"])


if __name__ == "__main__":
    main()
