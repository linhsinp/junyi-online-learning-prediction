# -*- coding: utf-8 -*-
"""Download data from Google Cloud Storage."""

import argparse
from scripts.gcs_utils import download_blob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--blob", type=str, required=True)
    parser.add_argument("--dest", type=str, default="data.csv")
    args = parser.parse_args()

    download_blob(args.bucket, args.blob, args.dest)


if __name__ == "__main__":
    main()
