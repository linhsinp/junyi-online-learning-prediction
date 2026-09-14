"""Curated event-partition transfers for local and GCS-backed preprocessing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path, PurePosixPath

from google.cloud import storage

from junyi_predictor.progress import operation, timed
from junyi_predictor.settings import DataLakeSettings


def _month_prefixes(start_date: datetime, end_date: datetime, prefix: str) -> list[str]:
    """Return month-partition prefixes overlapping a half-open input interval."""
    month = datetime(start_date.year, start_date.month, 1)
    prefixes = []
    while month < end_date:
        prefixes.append(
            f"{prefix.strip('/')}/year={month.year}/month={month.month:02d}"
        )
        month = (
            datetime(month.year + 1, 1, 1)
            if month.month == 12
            else datetime(month.year, month.month + 1, 1)
        )
    return prefixes


@timed("data_lake.upload", fields=("bucket_name", "prefix"))
def upload_curated_log(
    local_root: Path,
    bucket_name: str,
    prefix: str,
    *,
    client: storage.Client | None = None,
) -> int:
    """Upload an existing curated partition tree without altering local files."""
    if not local_root.is_dir():
        raise FileNotFoundError(f"Curated log root does not exist: {local_root}")
    bucket = (client or storage.Client()).bucket(bucket_name)
    files = [path for path in local_root.rglob("*") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No curated files found under {local_root}")
    with operation("data_lake.upload", bucket=bucket_name, prefix=prefix) as progress:
        for path in files:
            key = f"{prefix.strip('/')}/{path.relative_to(local_root).as_posix()}"
            bucket.blob(key).upload_from_filename(path)
        progress.update(file_count=len(files))
    return len(files)


@timed("data_lake.upload_dimensions", fields=("bucket_name", "prefix"))
def upload_dimension_csvs(
    user_path: Path,
    content_path: Path,
    bucket_name: str,
    prefix: str,
    *,
    client: storage.Client | None = None,
) -> None:
    """Stage the two Cloud SQL dimension inputs under a stable GCS prefix."""
    bucket = (client or storage.Client()).bucket(bucket_name)
    root_prefix = prefix.strip("/")
    for source in (user_path, content_path):
        if not source.is_file():
            raise FileNotFoundError(f"Dimension CSV does not exist: {source}")
        bucket.blob(f"{root_prefix}/{source.name}").upload_from_filename(source)


@timed("data_lake.download", fields=("bucket_name", "prefix"))
def download_curated_log_partitions(
    bucket_name: str,
    prefix: str,
    start_date: datetime,
    end_date: datetime,
    destination: Path,
    *,
    client: storage.Client | None = None,
) -> Path:
    """Download only monthly partitions needed by the preprocessing input window."""
    bucket = (client or storage.Client()).bucket(bucket_name)
    root_prefix = prefix.strip("/")
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    with operation(
        "data_lake.download", bucket=bucket_name, prefix=root_prefix
    ) as progress:
        for month_prefix in _month_prefixes(start_date, end_date, root_prefix):
            for blob in bucket.list_blobs(prefix=f"{month_prefix}/"):
                relative = PurePosixPath(blob.name).relative_to(root_prefix)
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(target)
                downloaded += 1
        progress.update(file_count=downloaded)
    if not downloaded:
        raise FileNotFoundError(
            f"No curated partitions found for {start_date.isoformat()} to {end_date.isoformat()}"
        )
    return destination


def resolve_curated_log_root(
    settings: DataLakeSettings,
    start_date: datetime,
    end_date: datetime,
    destination: Path,
) -> Path:
    """Return a local root, downloading the requested GCS partitions if needed."""
    if settings.backend == "local":
        return Path(settings.curated_log_root)
    if settings.backend == "gcs" and settings.gcs_bucket:
        return download_curated_log_partitions(
            settings.gcs_bucket,
            settings.gcs_prefix,
            start_date,
            end_date,
            destination,
        )
    raise ValueError(
        "GCS data lake requires DATA_LAKE_BACKEND=gcs and DATA_LAKE_BUCKET"
    )
