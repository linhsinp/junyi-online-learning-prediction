from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from junyi_predictor.contracts import PipelineRun, PreprocessedSnapshot


def test_pipeline_run_round_trips_as_a_flyte_safe_json_payload():
    run = PipelineRun(
        training_run_id="run-1",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    payload = run.model_dump(mode="json")

    assert PipelineRun.model_validate(payload) == run


def test_preprocessed_snapshot_rejects_invalid_durable_artifact_metadata():
    with pytest.raises(ValidationError):
        PreprocessedSnapshot(
            training_run_id="",
            log_uri="",
            user_uri="artifacts/user.parquet",
            content_uri="artifacts/content.parquet",
            row_count=-1,
        )


def test_legacy_run_id_payload_is_accepted_but_new_payload_uses_training_run_id():
    run = PipelineRun(
        run_id="legacy-run",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    payload = run.model_dump(mode="json")
    assert payload["training_run_id"] == "legacy-run"
    assert "run_id" not in payload
    legacy = {
        "run_id": "legacy-run",
        **{k: v for k, v in payload.items() if k != "training_run_id"},
    }
    assert PipelineRun.model_validate(legacy) == run
