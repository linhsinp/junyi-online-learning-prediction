from __future__ import annotations

import pandas as pd

from junyi_predictor.bootstrap.lake import materialize_log_parquet


def test_materialize_log_parquet_partitions_events_by_year_and_month(tmp_path):
    source = tmp_path / "log.csv"
    destination = tmp_path / "curated" / "log_problem"
    pd.DataFrame(
        {
            "timestamp_TW": ["2024-01-31 23:00:00 UTC", "2024-02-01 00:00:00 UTC"],
            "uuid": ["u1", "u2"],
            "ucid": ["c1", "c2"],
        }
    ).to_csv(source, index=False)

    materialize_log_parquet(source, destination, chunk_rows=1, target_rows_per_file=1)

    january = destination / "year=2024" / "month=01" / "part-00000.parquet"
    february = destination / "year=2024" / "month=02" / "part-00000.parquet"
    assert january.exists()
    assert february.exists()
    assert pd.read_parquet(january)["timestamp_TW"].iloc[0] == pd.Timestamp(
        "2024-01-31 23:00:00"
    )
