"""Context shared by nested log calls without leaking between executions."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_context: ContextVar[dict[str, Any]] = ContextVar("junyi_log_context", default={})


def current_context() -> dict[str, Any]:
    """Return a copy suitable for passing explicitly to a reporting thread."""
    return _context.get().copy()


@contextmanager
def bind_context(**fields: Any) -> Iterator[None]:
    """Extend the current context and restore it even after cancellation."""
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)
