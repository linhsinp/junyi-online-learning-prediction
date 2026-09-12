"""Application-owned, structured logging with safe optional file persistence."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from junyi_observability.context import current_context

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_SENSITIVE = re.compile(
    r"password|passwd|secret|token|credential|authorization|api.?key|"
    r"database_url|parameters|raw_data|input_value",
    re.I,
)
_URL_CREDENTIALS = re.compile(r"(\w+://)[^\s/@]+:[^\s/@]*@")
_AUTH_HEADER = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+")
_ASSIGNMENT = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_-]?key|authorization)"
    r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_LOCK = threading.RLock()


def _safe(value: Any, key: str = "") -> Any:
    if _SENSITIVE.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _safe(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, str):
        value = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", value)
        value = _AUTH_HEADER.sub(r"\1 [REDACTED]", value)
        return _ASSIGNMENT.sub(r"\1=[REDACTED]", value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (Path, datetime)):
        return _safe(str(value))
    # Do not call arbitrary reprs: dataframes and model objects can contain inputs.
    return f"<{type(value).__name__}>"


def _exception(error: BaseException) -> dict[str, Any]:
    """Retain stack locations without locals, source lines, or unsafe SDK payloads."""
    frames = [
        {"file": frame.filename, "line": frame.lineno, "function": frame.name}
        for frame in traceback.extract_tb(error.__traceback__)
    ]
    # SQLAlchemy messages contain SQL/parameters; Pydantic messages contain inputs.
    # Suppress the complete rendered message, including nested driver exceptions.
    modules = {cls.__module__.split(".")[0] for cls in type(error).__mro__}
    sensitive = isinstance(error, KeyError) or bool(
        modules & {"sqlalchemy", "pydantic", "pydantic_core", "psycopg2", "sqlite3"}
    )
    return {
        "type": type(error).__name__,
        "message": "[input-bearing exception message omitted]"
        if sensitive
        else _safe(str(error)),
        "traceback": frames,
    }


class StructuredFormatter(logging.Formatter):
    """Render identical event fields as JSON or readable single-line text."""

    def __init__(self, service: str, format: Literal["json", "text"] = "json"):
        super().__init__()
        self.service = service
        self.output_format = format

    def format(self, record: logging.LogRecord) -> str:
        fields = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        data = {
            **current_context(),
            **fields,
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "severity": record.levelname,
            "service": fields.get(
                "service", current_context().get("service", self.service)
            ),
            "logger": record.name,
            "event": fields.get("event", "application.message"),
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            data["exception"] = _exception(record.exc_info[1])
        data = _safe(data)
        if self.output_format == "json":
            return json.dumps(data, ensure_ascii=False, allow_nan=False)
        prefix = f"{data.pop('timestamp')} {data.pop('severity')} {data.pop('event')}"
        message = json.dumps(data.pop("message"), ensure_ascii=False)
        return f"{prefix} {message} " + " ".join(
            f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in data.items()
        )


def _warn(message: str) -> None:
    try:
        sys.stderr.write(f"Junyi logging warning: {message}\n")
        sys.stderr.flush()
    except Exception:
        pass


class _StreamHandler(logging.StreamHandler):
    def handleError(self, record: logging.LogRecord) -> None:
        # The default implementation can print the unsafe original record/exception.
        _warn("could not emit a log record")


class _FileHandler(RotatingFileHandler):
    failed: bool = False
    console_enabled: bool = True

    def emit(self, record: logging.LogRecord) -> None:
        if self.failed:
            self._fallback(record)
        else:
            super().emit(record)

    def _fallback(self, record: logging.LogRecord) -> None:
        if self.console_enabled:
            return
        handler = _StreamHandler()
        handler.setFormatter(self.formatter)
        handler.handle(record)

    def handleError(self, record: logging.LogRecord) -> None:
        if not self.failed:
            _warn("file logging failed; retaining stderr diagnostics")
        self.failed = True
        self._fallback(record)


@dataclass(frozen=True)
class LoggingOptions:
    """Explicit options supplied by an application, never read on import."""

    service: str
    namespaces: tuple[str, ...]
    level: str = "INFO"
    format: Literal["json", "text"] = "json"
    directory: Path | None = None
    console: bool = True


def configure_logging(options: LoggingOptions) -> Path | None:
    """Configure owned handlers once per process/options and return the log file."""
    if options.format not in ("json", "text"):
        raise ValueError("JUNYI_LOG_FORMAT must be json or text")
    level = logging.getLevelName(options.level.upper())
    if not isinstance(level, int):
        raise ValueError("JUNYI_LOG_LEVEL must be a standard logging level")
    if not options.namespaces or any(
        name in ("", "root") for name in options.namespaces
    ):
        raise ValueError("Explicit non-root logger namespaces are required")
    with _LOCK:
        loggers = [logging.getLogger(name) for name in options.namespaces]
        signature = (os.getpid(), options)
        owned = [h for h in loggers[0].handlers if getattr(h, "_junyi_owned", False)]
        if owned and all(
            getattr(h, "_junyi_signature", None) == signature for h in owned
        ):
            return next(
                (Path(h.baseFilename) for h in owned if isinstance(h, _FileHandler)),
                None,
            )
        for logger in loggers:
            for handler in logger.handlers[:]:
                if getattr(handler, "_junyi_owned", False):
                    logger.removeHandler(handler)
                    handler.close()
        handlers: list[logging.Handler] = []
        log_path = None
        console = options.console
        if options.directory is not None:
            try:
                options.directory.mkdir(parents=True, exist_ok=True)
                log_path = (
                    options.directory / f"junyi-{os.getpid()}-{uuid4().hex[:12]}.jsonl"
                )
                file_handler = _FileHandler(
                    log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
                )
                file_handler.console_enabled = console
                file_handler.setFormatter(StructuredFormatter(options.service))
                handlers.append(file_handler)
            except OSError:
                _warn("cannot open log file; retaining stderr diagnostics")
                log_path = None
                console = True
        if console or not handlers:
            stream = _StreamHandler()
            stream.setFormatter(StructuredFormatter(options.service, options.format))
            handlers.append(stream)
        for handler in handlers:
            handler._junyi_owned = True
            handler._junyi_signature = signature
        for logger in loggers:
            logger.setLevel(level)
            logger.propagate = False
            for handler in handlers:
                logger.addHandler(handler)
        return log_path
