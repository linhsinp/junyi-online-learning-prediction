"""Observe logs from a running CLI process, not just after task completion."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "logging_workflow.py"


def read_events(directory: Path) -> list[dict]:
    events = []
    for path in directory.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # A concurrent write can leave a partial final line until next poll.
                continue
    return events


@pytest.mark.parametrize("mode", ["success", "failure", "interrupt", "tui"])
def test_local_flyte_progress_and_terminal_outcomes(tmp_path, mode):
    fixture_path = tmp_path / "logging_workflow.py"
    shutil.copy2(FIXTURE, fixture_path)
    gate = tmp_path / "continue"
    log_dir = tmp_path / "logs"
    output_path = tmp_path / "console.txt"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "DATABASE_URL": f"sqlite:///{tmp_path / 'database.sqlite'}",
        "ARTIFACT_BACKEND": "local",
        "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "JUNYI_LOG_FORMAT": "text",
        "JUNYI_LOG_LEVEL": "INFO",
        "JUNYI_LOG_DIR": str(log_dir),
        "JUNYI_LOG_CONSOLE": "0" if mode == "tui" else "1",
        "JUNYI_LOG_INTERVAL_SECONDS": "0.05",
        "JUNYI_TEST_GATE": str(gate),
        "JUNYI_TEST_FAILURE": "1" if mode == "failure" else "0",
    }
    command = [str(Path(sys.executable).parent / "flyte"), "run", "--local"]
    if mode == "tui":
        command.append("--tui")
    command += [
        str(fixture_path),
        "training_pipeline",
        "--start_date",
        "2019-06-01T00:00:00",
        "--end_date",
        "2019-06-10T00:00:00",
        "--training_run_id",
        f"logging-{mode}-{tmp_path.name}",
    ]
    with output_path.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=tmp_path,
            env=env,
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                events = read_events(log_dir)
                if any(
                    e["event"] == "operation.running"
                    and e.get("operation") == "input.fixture"
                    for e in events
                ):
                    break
                if process.poll() is not None:
                    pytest.fail(output_path.read_text())
                time.sleep(0.02)
            else:
                pytest.fail("No live heartbeat: " + output_path.read_text())
            assert process.poll() is None
            assert not any(e["event"] == "execution.completed" for e in events)
            if mode == "interrupt":
                process.send_signal(signal.SIGINT)
            else:
                gate.touch()
            if mode == "tui":
                # The TUI intentionally stays open after completion until q.
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if any(
                        e["event"] == "execution.completed" and e["stage"] == "pipeline"
                        for e in read_events(log_dir)
                    ):
                        break
                    time.sleep(0.02)
                else:
                    pytest.fail(
                        "TUI workflow did not complete: " + output_path.read_text()
                    )
                process.stdin.write(b"q")
                process.stdin.flush()
            code = process.wait(timeout=20)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdin.close()
    events = read_events(log_dir)
    console = output_path.read_text()
    if mode in ("success", "tui"):
        assert code == 0, console
        assert any(e["event"] == "registration.approved" for e in events)
        fits = [
            e
            for e in events
            if e["event"] == "operation.completed" and e.get("operation") == "model.fit"
        ]
        assert len(fits) == 4
        assert (
            len(
                {
                    e["flyte_action_id"]
                    for e in events
                    if e["event"] == "execution.started"
                }
            )
            == 4
        )
        assert (tmp_path / "artifacts" / "models" / "approved.json").exists()
        assert ("operation.started" in console) == (mode != "tui")
    elif mode == "failure":
        assert code != 0
        failures = [
            e
            for e in events
            if e["event"] == "execution.failed" and e["stage"] == "preprocessing"
        ]
        assert failures and all(e["operation"] == "input.fixture" for e in failures)
        assert not any(e["event"] == "registration.approved" for e in events)
    else:
        assert code != 0
        assert any(e["event"] == "operation.running" for e in events)
        assert not any(
            e["event"] == "execution.completed" and e["stage"] == "pipeline"
            for e in events
        )


def test_loaded_module_bundle_contains_shared_package(tmp_path):
    # Exercise the installed Flyte bundler without uploading or touching local data.
    import asyncio

    from flyte._code_bundle import build_code_bundle

    from junyi_predictor.workflows.training import training_pipeline

    assert training_pipeline is not None
    bundle = asyncio.run(
        build_code_bundle(
            ROOT / "src",
            dryrun=True,
            copy_bundle_to=tmp_path,
            copy_style="loaded_modules",
        )
    )
    with tarfile.open(bundle.tgz) as archive:
        names = archive.getnames()
    assert any(name.endswith("junyi_observability/logging.py") for name in names)
    assert any(name.endswith("junyi_observability/context.py") for name in names)
    assert any(name.endswith("junyi_predictor/progress.py") for name in names)
