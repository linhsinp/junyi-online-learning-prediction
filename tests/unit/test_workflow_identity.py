from unittest.mock import Mock, patch

from junyi_predictor.workflows.training import _training_run_column


def test_new_output_tables_use_training_run_id():
    engine = Mock()
    with patch("junyi_predictor.workflows.training.inspect") as inspected:
        inspected.return_value.get_columns.return_value = []
        assert _training_run_column(engine, "processed_log") == "training_run_id"


def test_existing_legacy_output_tables_keep_pipeline_run_id():
    engine = Mock()
    with patch("junyi_predictor.workflows.training.inspect") as inspected:
        inspected.return_value.get_columns.return_value = [{"name": "pipeline_run_id"}]
        assert _training_run_column(engine, "processed_log") == "pipeline_run_id"


def test_partially_migrated_output_tables_prefer_training_run_id():
    engine = Mock()
    with patch("junyi_predictor.workflows.training.inspect") as inspected:
        inspected.return_value.get_columns.return_value = [
            {"name": "pipeline_run_id"},
            {"name": "training_run_id"},
        ]
        assert _training_run_column(engine, "processed_log") == "training_run_id"
