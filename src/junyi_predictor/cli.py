"""Small operational commands for local development and container smoke tests."""

from __future__ import annotations

import argparse
import os

from junyi_predictor.bootstrap.database import seed_database_from_raw_files
from junyi_predictor.bootstrap.kaggle import download_kaggle_data


def main() -> None:
    """Run an explicit local bootstrap command."""
    parser = argparse.ArgumentParser(prog="junyi-predictor")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("download-data")
    subcommands.add_parser("seed-db")
    args = parser.parse_args()

    if args.command == "download-data":
        download_kaggle_data()
    else:
        seed_database_from_raw_files(os.environ["DATABASE_URL"])


if __name__ == "__main__":
    main()
