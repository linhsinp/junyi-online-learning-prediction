"""Training-owned operation reporting and executable error boundaries."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar
from uuid import uuid4

from junyi_observability import (
    LoggingOptions,
    bind_context,
    configure_logging,
    current_context,
)

P = ParamSpec("P")
R = TypeVar("R")
logger = logging.getLogger(__name__)
_failure: ContextVar[dict[str, Any] | None] = ContextVar("junyi_failure", default=None)
_active: ContextVar[Progress | None] = ContextVar("junyi_operation", default=None)
LOG_ENV_DEFAULTS = {
    "JUNYI_LOG_LEVEL": "INFO",
    "JUNYI_LOG_FORMAT": "json",
    "JUNYI_LOG_INTERVAL_SECONDS": "60",
    "JUNYI_LOG_CONSOLE": "1",
}


def remote_logging_env() -> dict[str, str]:
    """Set remote defaults without baking local file paths into task containers."""
    return LOG_ENV_DEFAULTS.copy()


def _interval() -> float:
    value = float(os.getenv("JUNYI_LOG_INTERVAL_SECONDS", "60"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("JUNYI_LOG_INTERVAL_SECONDS must be positive and finite")
    return value


class Progress:
    """Report thread-safe counters; a heartbeat alone does not imply progress."""

    def __init__(
        self, log: logging.Logger, clock: Callable[[], float] = time.monotonic
    ):
        self.log = log
        self.clock = clock
        self.started = clock()
        self.fields: dict[str, Any] = {}
        self.lock = threading.Lock()
        self.suspended = False

    def update(self, **fields: Any) -> None:
        with self.lock:
            self.fields.update(fields)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            fields = self.fields.copy()
        elapsed = max(0.0, self.clock() - self.started)
        fields["elapsed_seconds"] = round(elapsed, 3)
        if "completed_rows" in fields and elapsed:
            fields["rows_per_second"] = round(fields["completed_rows"] / elapsed, 2)
        return fields

    def report(self) -> None:
        if not self.suspended:
            self.log.info(
                "Operation still running",
                extra={"event": "operation.running", **self.snapshot()},
            )


@contextmanager
def operation(
    name: str,
    *,
    log: logging.Logger = logger,
    heartbeat: bool = True,
    clock: Callable[[], float] = time.monotonic,
    **fields: Any,
) -> Iterator[Progress]:
    """Time an operation, preserving its context for the outer error boundary."""
    progress = Progress(log, clock)
    progress.update(**fields)
    parent = _active.get()
    if parent:
        parent.suspended = True
    token = _active.set(progress)
    stop = threading.Event()
    thread = None
    try:
        with bind_context(operation=name, **fields):
            log.info("Operation started", extra={"event": "operation.started"})
            try:
                if heartbeat and log.isEnabledFor(logging.INFO):
                    interval = _interval()
                    context = current_context()

                    def report() -> None:
                        with bind_context(**context):
                            while not stop.wait(interval):
                                progress.report()

                    thread = threading.Thread(
                        target=report, name="junyi-progress", daemon=True
                    )
                    try:
                        thread.start()
                    except RuntimeError:
                        thread = None
                        log.warning(
                            "Heartbeat reporter unavailable",
                            extra={"event": "logging.reporter_unavailable"},
                        )
                yield progress
            except BaseException:
                failure = _failure.get()
                if failure is not None and not failure:
                    snapshot = progress.snapshot()
                    failure.update(
                        {
                            **current_context(),
                            **snapshot,
                            "operation_elapsed_seconds": snapshot["elapsed_seconds"],
                        }
                    )
                raise
            else:
                # Stop before completion so the final event is never a heartbeat.
                stop.set()
                if thread:
                    thread.join(timeout=1)
                with progress.lock:
                    progress.suspended = True
                log.info(
                    "Operation completed",
                    extra={"event": "operation.completed", **progress.snapshot()},
                )
    finally:
        stop.set()
        if thread:
            thread.join(timeout=1)
        _active.reset(token)
        if parent:
            parent.suspended = False


def timed(
    name: str, *, heartbeat: bool = True, fields: tuple[str, ...] = ()
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Instrument a synchronous helper without altering its arguments or result."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            metadata = {}
            if fields:
                arguments = inspect.signature(function).bind(*args, **kwargs)
                arguments.apply_defaults()
                metadata = {key: arguments.arguments[key] for key in fields}
            with operation(
                name,
                log=logging.getLogger(function.__module__),
                heartbeat=heartbeat,
                **metadata,
            ):
                return function(*args, **kwargs)

        return wrapped

    return decorate


@contextmanager
def execution(
    stage: str,
    *,
    service: str = "junyi-training",
    training_run_id: str | None = None,
    summary_only: bool = False,
    **fields: Any,
) -> Iterator[None]:
    """Configure first, then log a single failure and preserve original exceptions."""
    # Establish fallback logging before parsing potentially invalid log settings.
    fallback = LoggingOptions(service=service, namespaces=("junyi_predictor",))
    # Reuse valid existing handlers so repeated local task entry doesn't reopen files.
    if not logging.getLogger("junyi_predictor").handlers:
        configure_logging(fallback)
    started = time.monotonic()
    token = _failure.set({})
    with bind_context(
        service=service,
        stage=stage,
        training_run_id=training_run_id,
        **fields,
    ):
        try:
            options = LoggingOptions(
                service="junyi",
                namespaces=("junyi_predictor",),
                level=os.getenv("JUNYI_LOG_LEVEL", "INFO"),
                format=os.getenv("JUNYI_LOG_FORMAT", "json"),
                directory=Path(os.environ["JUNYI_LOG_DIR"])
                if os.getenv("JUNYI_LOG_DIR")
                else None,
                console=os.getenv("JUNYI_LOG_CONSOLE", "1") != "0",
            )
            path = configure_logging(options)
            _interval()
            logger.info(
                "Execution started",
                extra={
                    "event": "execution.started",
                    "log_file": str(path) if path else None,
                },
            )
            yield
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.warning(
                "Execution interrupted",
                extra={
                    "event": "execution.interrupted",
                    **(_failure.get() or {}),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
            raise
        except Exception:
            failure = _failure.get() or {}
            child_failure = summary_only and str(
                failure.get("operation", "")
            ).startswith("pipeline.")
            logger.error(
                "Execution failed",
                exc_info=not child_failure,
                extra={
                    "event": "execution.failed",
                    **(_failure.get() or {}),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
            raise
        else:
            logger.info(
                "Execution completed",
                extra={
                    "event": "execution.completed",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
        finally:
            _failure.reset(token)


def task_logging(
    stage: str,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Attach a fresh logging scope to each Flyte invocation, including retries."""

    def decorate(function: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(function)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            import flyte

            arguments = inspect.signature(function).bind(*args, **kwargs)
            arguments.apply_defaults()
            payload = next(
                (v for v in arguments.arguments.values() if isinstance(v, dict)), {}
            )
            training_run_id = payload.get("training_run_id", payload.get("run_id"))
            if stage == "pipeline":
                training_run_id = (
                    arguments.arguments.get("training_run_id")
                    or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                    + "-"
                    + uuid4().hex[:8]
                )
                arguments.arguments["training_run_id"] = training_run_id
            if not isinstance(training_run_id, str):
                training_run_id = None
            context = flyte.ctx()
            identifiers = {}
            if context is not None:
                action = context.action
                identifiers = {
                    "flyte_run_id": action.run_name,
                    "flyte_action_id": action.name,
                }
            with execution(
                stage,
                training_run_id=training_run_id,
                summary_only=stage == "pipeline",
                **identifiers,
            ):
                return await function(*arguments.args, **arguments.kwargs)

        return wrapped

    return decorate
