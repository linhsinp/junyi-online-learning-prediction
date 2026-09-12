"""Local data-lake materialization for training event history."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from junyi_predictor.paths import CURATED_LOG_DIR, LOG_FILE
from junyi_predictor.progress import timed

LOG_DTYPES = {
    "uuid": "string",
    "ucid": "string",
    "upid": "string",
    "problem_number": "int16",
    "exercise_problem_repeat_session": "int16",
    "is_correct": "boolean",
    "total_sec_taken": "int16",
    "total_attempt_cnt": "int16",
    "used_hint_cnt": "int16",
    "is_hint_used": "boolean",
    "is_downgrade": "boolean",
    "is_upgrade": "boolean",
    "level": "int8",
}


def _partitioned_log_chunks(
    source: Path, chunk_rows: int
) -> Iterator[tuple[tuple[int, int], pd.DataFrame]]:
    for chunk in pd.read_csv(source, chunksize=chunk_rows, dtype=LOG_DTYPES):
        timestamp = pd.to_datetime(chunk["timestamp_TW"], utc=True).dt.tz_convert(None)
        chunk = chunk.assign(
            timestamp_TW=timestamp, year=timestamp.dt.year, month=timestamp.dt.month
        )
        for (year, month), partition in chunk.groupby(["year", "month"], sort=True):
            yield (int(year), int(month)), partition.drop(columns=["year", "month"])


@timed("bootstrap.materialize_parquet")
def materialize_log_parquet(
    source: Path = LOG_FILE,
    destination: Path = CURATED_LOG_DIR,
    chunk_rows: int = 250_000,
    target_rows_per_file: int = 1_500_000,
) -> None:
    """Convert raw events into year/month-partitioned Parquet files."""
    if destination.exists():
        shutil.rmtree(destination)
    writers: dict[tuple[int, int], tuple[pq.ParquetWriter, int, int]] = {}
    try:
        for partition_key, frame in _partitioned_log_chunks(source, chunk_rows):
            year, month = partition_key
            partition_dir = destination / f"year={year}" / f"month={month:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            writer, row_count, file_index = writers.get(partition_key, (None, 0, 0))
            if writer is None or row_count + len(frame) > target_rows_per_file:
                if writer is not None:
                    writer.close()
                    file_index += 1
                path = partition_dir / f"part-{file_index:05d}.parquet"
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                row_count = 0
            writer.write_table(table)
            writers[partition_key] = (writer, row_count + len(frame), file_index)
    finally:
        for writer, _, _ in writers.values():
            writer.close()
