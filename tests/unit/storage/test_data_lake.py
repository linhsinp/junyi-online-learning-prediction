from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from junyi_predictor.settings import DataLakeSettings
from junyi_predictor.storage.data_lake import (
    download_curated_log_partitions,
    resolve_curated_log_root,
    upload_curated_log,
    upload_dimension_csvs,
)


def test_upload_curated_log_preserves_partition_relative_paths(tmp_path: Path):
    root = tmp_path / "curated"
    source = root / "year=2024" / "month=06" / "part-00000.parquet"
    source.parent.mkdir(parents=True)
    source.write_text("data")
    bucket = MagicMock()
    client = MagicMock()
    client.bucket.return_value = bucket

    assert (
        upload_curated_log(root, "lake", "data/curated/log_problem", client=client) == 1
    )

    bucket.blob.assert_called_once_with(
        "data/curated/log_problem/year=2024/month=06/part-00000.parquet"
    )
    bucket.blob.return_value.upload_from_filename.assert_called_once_with(source)


def test_upload_dimension_csvs_uses_stable_gcs_names(tmp_path: Path):
    user = tmp_path / "Info_UserData.csv"
    content = tmp_path / "Info_Content.csv"
    user.write_text("uuid")
    content.write_text("ucid")
    bucket = MagicMock()
    client = MagicMock()
    client.bucket.return_value = bucket

    upload_dimension_csvs(user, content, "lake", "data/dimensions", client=client)

    assert [call.args[0] for call in bucket.blob.call_args_list] == [
        "data/dimensions/Info_UserData.csv",
        "data/dimensions/Info_Content.csv",
    ]


def test_download_curated_log_partitions_selects_required_months(tmp_path: Path):
    downloads: list[Path] = []

    class Blob:
        def __init__(self, name: str):
            self.name = name

        def download_to_filename(self, destination: Path) -> None:
            destination = Path(destination)
            downloads.append(destination)
            destination.write_text(self.name)

    bucket = MagicMock()
    bucket.list_blobs.side_effect = [
        [Blob("data/curated/log_problem/year=2024/month=06/part-00000.parquet")],
        [Blob("data/curated/log_problem/year=2024/month=07/part-00000.parquet")],
    ]
    client = MagicMock()
    client.bucket.return_value = bucket

    root = download_curated_log_partitions(
        "lake",
        "data/curated/log_problem",
        datetime(2024, 6, 10),
        datetime(2024, 7, 2),
        tmp_path,
        client=client,
    )

    assert root == tmp_path
    assert downloads == [
        tmp_path / "year=2024" / "month=06" / "part-00000.parquet",
        tmp_path / "year=2024" / "month=07" / "part-00000.parquet",
    ]
    assert [call.kwargs["prefix"] for call in bucket.list_blobs.call_args_list] == [
        "data/curated/log_problem/year=2024/month=06/",
        "data/curated/log_problem/year=2024/month=07/",
    ]


def test_resolve_curated_log_root_requires_a_bucket_for_gcs(tmp_path: Path):
    settings = DataLakeSettings("gcs", "unused", None, "data/curated/log_problem")

    with pytest.raises(ValueError, match="DATA_LAKE_BUCKET"):
        resolve_curated_log_root(
            settings, datetime(2024, 6, 1), datetime(2024, 6, 2), tmp_path
        )
