"""Ensure loguru import succeeds even in minimal dev environments."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from types import SimpleNamespace


def ensure_loguru() -> None:
    try:  # pragma: no cover - best effort import
        import loguru  # type: ignore

        if hasattr(loguru, "logger"):
            return
    except Exception:
        pass

    class _StubMessage:
        def __init__(self, record):
            self.record = record

    _CONTEXT_EXTRA: ContextVar[dict] = ContextVar("stub_loguru_context_extra", default={})

    class _StubLogger:
        def __init__(self, sinks=None, extra=None, *, lazy=False):
            self._sinks = sinks if sinks is not None else []
            self._extra = extra or {}
            self._lazy = bool(lazy)

        def bind(self, **kwargs):
            merged = dict(_CONTEXT_EXTRA.get())
            merged.update(self._extra)
            merged.update({k: v for k, v in kwargs.items() if v is not None})
            return _StubLogger(self._sinks, merged, lazy=self._lazy)

        def opt(self, **kwargs):
            return _StubLogger(self._sinks, self._extra, lazy=kwargs.get("lazy", self._lazy))

        @contextmanager
        def contextualize(self, **kwargs):
            current = dict(_CONTEXT_EXTRA.get())
            current.update({k: v for k, v in kwargs.items() if v is not None})
            token = _CONTEXT_EXTRA.set(current)
            try:
                yield self
            finally:
                _CONTEXT_EXTRA.reset(token)

        def add(self, sink, **kwargs):
            self._sinks.append(sink)
            return len(self._sinks)

        def log(self, level, message, *args, **kwargs):
            if self._lazy:
                args = tuple(item() if callable(item) else item for item in args)
                kwargs = {
                    key: value() if callable(value) else value
                    for key, value in kwargs.items()
                }
            formatted = message.format(*args, **kwargs) if (args or kwargs) else message
            extra = dict(_CONTEXT_EXTRA.get())
            extra.update(self._extra)
            record = {
                "level": {"name": level},
                "message": formatted,
                "extra": extra,
                "time": datetime.now(timezone.utc),
                "file": {"name": ""},
                "function": "",
                "line": 0,
            }
            payload = _StubMessage(record=record)
            for sink in list(self._sinks):
                try:
                    sink(payload)
                except Exception:
                    continue
            return True

        def debug(self, message, *args, **kwargs):
            return self.log("DEBUG", message, *args, **kwargs)

        def info(self, message, *args, **kwargs):
            return self.log("INFO", message, *args, **kwargs)

        def warning(self, message, *args, **kwargs):
            return self.log("WARNING", message, *args, **kwargs)

        def error(self, message, *args, **kwargs):
            return self.log("ERROR", message, *args, **kwargs)

        def exception(self, message, *args, **kwargs):
            return self.log("ERROR", message, *args, **kwargs)

        def __getattr__(self, name):  # pragma: no cover
            return self.log

    sys.modules["loguru"] = SimpleNamespace(logger=_StubLogger())
