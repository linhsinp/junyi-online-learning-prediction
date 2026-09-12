"""Isolate application logging state, which deliberately survives task calls."""

import json
import logging

import pytest


@pytest.fixture(autouse=True)
def logging_state(monkeypatch):
    for key in (
        "JUNYI_LOG_DIR",
        "JUNYI_LOG_LEVEL",
        "JUNYI_LOG_FORMAT",
        "JUNYI_LOG_CONSOLE",
        "JUNYI_LOG_INTERVAL_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    saved = []
    for name in ("junyi_predictor", "junyi_test"):
        logger = logging.getLogger(name)
        saved.append((logger, logger.handlers[:], logger.level, logger.propagate))
        logger.handlers = []
    yield
    for logger, handlers, level, propagate in saved:
        for handler in logger.handlers:
            if handler not in handlers:
                handler.close()
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate


@pytest.fixture
def records(tmp_path, monkeypatch):
    monkeypatch.setenv("JUNYI_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("JUNYI_LOG_CONSOLE", "0")

    def read():
        return [
            json.loads(line)
            for path in tmp_path.glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]

    return read
