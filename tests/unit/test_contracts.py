from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from junyi_predictor.contracts import PipelineRun, PreprocessedSnapshot


def test_pipeline_run_round_trips_as_a_flyte_safe_json_payload():
    run = PipelineRun(
        run_id="run-1",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    payload = run.model_dump(mode="json")

    assert PipelineRun.model_validate(payload) == run


def test_preprocessed_snapshot_rejects_invalid_durable_artifact_metadata():
    with pytest.raises(ValidationError):
        PreprocessedSnapshot(
            run_id="",
            log_uri="",
            user_uri="artifacts/user.parquet",
            content_uri="artifacts/content.parquet",
            row_count=-1,
        )
