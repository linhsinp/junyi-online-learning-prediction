import asyncio
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import StatementError

from junyi_observability import (
    LoggingOptions,
    bind_context,
    configure_logging,
    current_context,
)
from junyi_observability.logging import (
    StructuredFormatter,
    _FileHandler,
    _StreamHandler,
    _warn,
)


def test_json_fields_and_configuration_are_idempotent(tmp_path, capsys):
    options = LoggingOptions("test-service", ("junyi_test",), directory=tmp_path)
    root_handlers = logging.getLogger().handlers[:]
    flyte_handlers = logging.getLogger("flyte").handlers[:]
    path = configure_logging(options)
    assert configure_logging(options) == path
    with bind_context(run_id="run-1", stage="features"):
        logging.getLogger("junyi_test.features").info(
            "Building",
            extra={"event": "features.started", "row_count": 12, "shape": (12, 2)},
        )
    persisted = [json.loads(line) for line in path.read_text().splitlines()]
    console = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert persisted == console
    assert len(persisted) == 1
    assert persisted[0] | {"timestamp": "ignored"} == {
        "timestamp": "ignored",
        "service": "test-service",
        "severity": "INFO",
        "logger": "junyi_test.features",
        "event": "features.started",
        "message": "Building",
        "run_id": "run-1",
        "stage": "features",
        "row_count": 12,
        "shape": [12, 2],
    }
    assert (
        datetime.fromisoformat(persisted[0]["timestamp"]).utcoffset().total_seconds()
        == 0
    )
    assert logging.getLogger().handlers == root_handlers
    assert logging.getLogger("flyte").handlers == flyte_handlers


def test_text_console_json_file_and_level(tmp_path, capsys):
    configure_logging(
        LoggingOptions("test", ("junyi_test",), format="text", directory=tmp_path)
    )
    log = logging.getLogger("junyi_test.module")
    log.debug("hidden")
    log.info("first\nsecond", extra={"event": "operation.started", "completed_rows": 3})
    line = capsys.readouterr().err
    assert len(line.splitlines()) == 1
    assert "operation.started" in line and "completed_rows=3" in line
    assert "hidden" not in line
    assert (
        json.loads(next(tmp_path.glob("*.jsonl")).read_text())["message"]
        == "first\nsecond"
    )


def test_context_restores_after_errors_and_isolates_coroutines():
    async def worker(run_id):
        with bind_context(run_id=run_id):
            await asyncio.sleep(0)
            with pytest.raises(RuntimeError), bind_context(stage="nested"):
                raise RuntimeError("test")
            return current_context()

    async def run():
        return await asyncio.gather(worker("first"), worker("second"))

    assert asyncio.run(run()) == [{"run_id": "first"}, {"run_id": "second"}]
    assert current_context() == {}


def test_redaction_and_safe_non_json_values(tmp_path):
    path = configure_logging(
        LoggingOptions("test", ("junyi_test",), directory=tmp_path, console=False)
    )
    logging.getLogger("junyi_test").info(
        "url=postgresql://user:secret@host/db password=abc token='hidden value' Authorization: Bearer private-token",
        extra={
            "details": {
                "api_key": "private",
                "items": [
                    Path("artifact"),
                    float("nan"),
                    object(),
                    datetime(2020, 1, 1, tzinfo=timezone.utc),
                ],
            }
        },
    )
    text = path.read_text()
    assert all(
        value not in text for value in ("user:secret", "abc", "hidden value", "private")
    )
    data = json.loads(text)
    assert data["details"]["items"] == [
        "artifact",
        "nan",
        "<object>",
        "2020-01-01 00:00:00+00:00",
    ]


@pytest.mark.parametrize("kind", ["sql", "validation", "generic"])
def test_safe_exception_tracebacks(kind, tmp_path):
    path = configure_logging(
        LoggingOptions("test", ("junyi_test",), directory=tmp_path, console=False)
    )
    try:
        if kind == "sql":
            raise StatementError(
                "learner-secret",
                "SELECT learner-secret",
                {"input": "learner-secret"},
                ValueError("learner-secret"),
            )
        if kind == "validation":

            class Model(BaseModel):
                count: int

            Model(count="learner-secret")
        raise ValueError("password=learner-secret")
    except (StatementError, ValidationError, ValueError):
        logging.getLogger("junyi_test").exception(
            "Operation failed", extra={"event": "execution.failed"}
        )
    text = path.read_text()
    assert "learner-secret" not in text
    data = json.loads(text)
    assert any(
        frame["function"] == "test_safe_exception_tracebacks"
        for frame in data["exception"]["traceback"]
    )
    assert set(data["exception"]["traceback"][-1]) == {"file", "line", "function"}


def test_file_creation_failure_falls_back_to_stderr(tmp_path, capsys):
    file = tmp_path / "not-a-directory"
    file.write_text("existing")
    assert (
        configure_logging(
            LoggingOptions("test", ("junyi_test",), directory=file, console=False)
        )
        is None
    )
    logging.getLogger("junyi_test").warning("Still visible")
    output = capsys.readouterr().err
    assert "cannot open log file" in output and "Still visible" in output


@pytest.mark.parametrize("console", [True, False])
def test_file_write_failure_warns_once_and_retains_records(tmp_path, capsys, console):
    configure_logging(
        LoggingOptions("test", ("junyi_test",), directory=tmp_path, console=console)
    )
    handler = next(
        h
        for h in logging.getLogger("junyi_test").handlers
        if isinstance(h, _FileHandler)
    )
    with patch.object(handler, "shouldRollover", side_effect=OSError("secret")):
        logging.getLogger("junyi_test").warning("First")
        logging.getLogger("junyi_test").warning("Second")
    output = capsys.readouterr().err
    assert output.count("file logging failed") == 1
    assert output.count('"message": "First"') == 1
    assert output.count('"message": "Second"') == 1
    assert "secret" not in output


def test_rotation_and_reconfiguration_preserve_unowned_handlers(tmp_path):
    log = logging.getLogger("junyi_test")
    unowned = logging.NullHandler()
    log.addHandler(unowned)
    options = LoggingOptions("test", ("junyi_test",), directory=tmp_path, console=False)
    path = configure_logging(options)
    handler = next(h for h in log.handlers if isinstance(h, _FileHandler))
    assert handler.maxBytes == 10 * 1024 * 1024 and handler.backupCount == 3
    handler.maxBytes = 256
    for number in range(10):
        log.info("record %s", number)
    assert Path(str(path) + ".3").exists()
    assert not Path(str(path) + ".4").exists()
    configure_logging(LoggingOptions("test", ("junyi_test",), level="WARNING"))
    assert handler.stream is None and unowned in log.handlers


@pytest.mark.parametrize(
    "changes",
    [
        {"level": "bogus"},
        {"format": "bogus"},
        {"namespaces": ("",)},
        {"namespaces": ()},
    ],
)
def test_invalid_configuration(changes):
    args = {"service": "test", "namespaces": ("junyi_test",), **changes}
    with pytest.raises(ValueError):
        configure_logging(LoggingOptions(**args))


def test_disabled_console_without_file_still_has_diagnostics(capsys):
    configure_logging(LoggingOptions("test", ("junyi_test",), console=False))
    logging.getLogger("junyi_test").error("Visible")
    assert "Visible" in capsys.readouterr().err


def test_stream_errors_do_not_expose_original_record(monkeypatch):
    stream = io.StringIO()
    handler = _StreamHandler(stream)
    handler.setFormatter(StructuredFormatter("test"))
    stream.close()
    handler.handle(logging.makeLogRecord({"msg": "password=private"}))
    # Even a broken stderr must not mask the application's original exception.
    monkeypatch.setattr("sys.stderr", stream)
    _warn("cannot log")
