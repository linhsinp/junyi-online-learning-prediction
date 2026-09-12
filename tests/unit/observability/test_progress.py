import asyncio
import logging
import threading
from types import SimpleNamespace

import pytest

from junyi_observability import current_context
from junyi_predictor.progress import (
    Progress,
    execution,
    operation,
    remote_logging_env,
    task_logging,
    timed,
)


def test_nested_failure_keeps_innermost_operation_and_original_exception(records):
    error = RuntimeError("original")
    with pytest.raises(RuntimeError) as caught:
        with execution("features", run_id="run-1"):
            with operation("outer", heartbeat=False):
                with operation("inner", heartbeat=False, model_type="example"):
                    raise error
    assert caught.value is error
    failures = [r for r in records() if r["event"] == "execution.failed"]
    assert len(failures) == 1
    assert failures[0]["operation"] == "inner"
    assert failures[0]["model_type"] == "example"
    assert failures[0]["exception"]["type"] == "RuntimeError"
    assert current_context() == {}


@pytest.mark.parametrize("error", [KeyboardInterrupt(), asyncio.CancelledError()])
def test_cancellation_cleans_up_threads_and_preserves_exception(records, error):
    with pytest.raises(type(error)) as caught:
        with execution("training"):
            with operation("model.fit"):
                raise error
    assert caught.value is error
    assert any(r["event"] == "execution.interrupted" for r in records())
    assert not any(r["event"] == "execution.failed" for r in records())
    assert not any(t.name == "junyi-progress" for t in threading.enumerate())


def test_heartbeat_while_blocked_has_context_and_stops(records, monkeypatch):
    monkeypatch.setenv("JUNYI_LOG_INTERVAL_SECONDS", "0.01")
    emitted = threading.Event()
    real_report = Progress.report

    def report(self):
        real_report(self)
        emitted.set()

    monkeypatch.setattr(Progress, "report", report)
    with execution("training", run_id="heartbeat-run"):
        with operation("model.fit", model_type="tree") as progress:
            progress.update(completed_rows=1000, total_rows=3000)
            assert emitted.wait(2), "reporter did not run while main thread was blocked"
    events = records()
    heartbeat = next(r for r in events if r["event"] == "operation.running")
    assert heartbeat["run_id"] == "heartbeat-run"
    assert heartbeat["model_type"] == "tree" and heartbeat["completed_rows"] == 1000
    assert events[-1]["event"] == "execution.completed"
    assert not any(t.name == "junyi-progress" for t in threading.enumerate())


def test_timing_progress_and_parent_suspension(records):
    clock_value = [10.0]
    with execution("features"):
        with operation(
            "parent", clock=lambda: clock_value[0], heartbeat=False
        ) as parent:
            clock_value[0] = 12.0
            parent.update(completed_rows=100, total_rows=200)
            assert parent.snapshot()["rows_per_second"] == 50
            with operation("child", heartbeat=False):
                assert parent.suspended
                parent.report()
            assert not parent.suspended
            parent.report()
    assert sum(r["event"] == "operation.running" for r in records()) == 1
    assert (
        next(
            r
            for r in records()
            if r["event"] == "operation.completed" and r["operation"] == "parent"
        )["elapsed_seconds"]
        == 2
    )


def test_timed_helper_preserves_result_and_only_selected_arguments(records):
    @timed("artifact.write", fields=("key",), heartbeat=False)
    def write(key: str, payload: dict) -> str:
        return key

    write.__wrapped__.__module__ = "junyi_predictor.test_helper"
    with execution("training"):
        assert write("model.json", {"learner": "learner-secret"}) == "model.json"
    assert any(r.get("key") == "model.json" for r in records())
    assert "learner-secret" not in str(records())


@pytest.mark.parametrize(
    "setting,value",
    [
        ("JUNYI_LOG_LEVEL", "bogus"),
        ("JUNYI_LOG_FORMAT", "bogus"),
        ("JUNYI_LOG_INTERVAL_SECONDS", "nan"),
        ("JUNYI_LOG_INTERVAL_SECONDS", "0"),
    ],
)
def test_configuration_failure_is_observable(setting, value, monkeypatch, capsys):
    monkeypatch.setenv(setting, value)
    with pytest.raises(ValueError), execution("startup"):
        pytest.fail("invalid settings allowed execution")
    assert "execution.failed" in capsys.readouterr().err


def test_task_invocations_have_distinct_ids_and_flyte_context(records, monkeypatch):
    monkeypatch.setattr(
        "flyte.ctx",
        lambda: SimpleNamespace(
            action=SimpleNamespace(run_name="flyte-run", name="a1")
        ),
    )

    @task_logging("features")
    async def task(payload: dict) -> dict:
        return payload

    payload = {"run_id": "same-run"}
    assert asyncio.run(task(payload)) == payload
    assert asyncio.run(task(payload)) == payload
    events = [r for r in records() if r["event"] == "execution.started"]
    assert events[0]["invocation_id"] != events[1]["invocation_id"]
    assert all(
        r["flyte_action_id"] == "a1" and r["run_id"] == "same-run" for r in events
    )


def test_pipeline_assigns_run_id_and_only_summarizes_child_failure(
    records, monkeypatch
):
    monkeypatch.setattr("flyte.ctx", lambda: None)

    @task_logging("features")
    async def child(payload: dict) -> dict:
        raise ValueError("failed")

    @task_logging("pipeline")
    async def parent(run_id: str = "") -> dict:
        assert run_id
        with operation("pipeline.features", heartbeat=False):
            return await child({"run_id": run_id})

    with pytest.raises(ValueError):
        asyncio.run(parent())
    failures = [r for r in records() if r["event"] == "execution.failed"]
    assert len(failures) == 2 and sum("exception" in r for r in failures) == 1
    assert failures[0]["run_id"] == failures[1]["run_id"]


def test_invalid_payload_does_not_log_inputs(records, monkeypatch):
    monkeypatch.setattr("flyte.ctx", lambda: None)

    @task_logging("features")
    async def task(payload: dict) -> dict:
        return {}

    asyncio.run(task({"run_id": ["learner-secret"]}))
    assert all(r["run_id"] is None for r in records())
    assert "learner-secret" not in str(records())


def test_remote_settings_do_not_embed_local_directory(monkeypatch):
    monkeypatch.setenv("JUNYI_LOG_DIR", "private/local")
    assert "JUNYI_LOG_DIR" not in remote_logging_env()
    assert remote_logging_env()["JUNYI_LOG_FORMAT"] == "json"


def test_progress_disabled_at_warning_level(records, monkeypatch):
    monkeypatch.setenv("JUNYI_LOG_LEVEL", "WARNING")
    with execution("training"), operation("fit"):
        assert not any(t.name == "junyi-progress" for t in threading.enumerate())
        logging.getLogger("junyi_predictor").warning("Diagnostic")
    assert len(records()) == 1


def test_parent_startup_error_has_traceback(records):
    with pytest.raises(ValueError), execution("pipeline", summary_only=True):
        raise ValueError("Invalid run settings")
    assert records()[-1]["exception"]["type"] == "ValueError"


def test_reporter_start_failure_does_not_stop_work(records, monkeypatch):
    def fail_start(self):
        raise RuntimeError("thread limit reached")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with execution("training"), operation("model.fit"):
        pass
    assert any(r["event"] == "logging.reporter_unavailable" for r in records())
    assert records()[-1]["event"] == "execution.completed"
