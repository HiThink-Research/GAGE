"""Ensure loguru import succeeds even in minimal dev environments."""

from __future__ import annotations

import sys
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

    class _StubLogger:
        def __init__(self, sinks=None, extra=None):
            self._sinks = sinks if sinks is not None else []
            self._extra = extra or {}

        def bind(self, **kwargs):
            merged = dict(self._extra)
            merged.update({k: v for k, v in kwargs.items() if v is not None})
            return _StubLogger(self._sinks, merged)

        def add(self, sink, **kwargs):
            self._sinks.append(sink)
            return len(self._sinks)

        def log(self, level, message, *args, **kwargs):
            formatted = message.format(*args, **kwargs) if (args or kwargs) else message
            record = {
                "level": {"name": level},
                "message": formatted,
                "extra": dict(self._extra),
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

        def __getattr__(self, name):  # pragma: no cover
            return self.log

    sys.modules["loguru"] = SimpleNamespace(logger=_StubLogger())
