from __future__ import annotations

import pytest

from gage_eval.observability.config import ObservabilityConfig
from gage_eval.observability.logger import ObservableLogger


class _TraceProbe:
    run_id = "trace-probe"

    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str | None]] = []

    def accepts_new_events(self) -> bool:
        return True

    def emit(self, event: str, payload: dict, sample_id: str | None = None) -> None:
        self.events.append((event, payload, sample_id))


class _DummyLogger:
    def bind(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return self

    def log(self, level, message, *args, **kwargs):  # noqa: ANN001, ANN003
        _ = level, message, args, kwargs
        return None


@pytest.mark.fast
def test_logger_does_not_emit_log_directly_when_sink_is_inactive(monkeypatch) -> None:
    trace = _TraceProbe()
    observable_logger = ObservableLogger(config=ObservabilityConfig(enabled=True))
    monkeypatch.setattr("gage_eval.observability.logger.base_logger", _DummyLogger())
    monkeypatch.setattr("gage_eval.observability.logger.is_log_sink_active", lambda: False)

    observable_logger.info("stage", "hello {name}", name="unit", trace=trace, sample_id="sample-1")

    assert trace.events == []


@pytest.mark.fast
def test_default_logger_does_not_expose_drain_buffer() -> None:
    observable_logger = ObservableLogger(config=ObservabilityConfig(enabled=True))

    observable_logger.info("stage", "hello")

    assert not hasattr(observable_logger, "drain_buffer")
    assert observable_logger.debug_buffer is None


@pytest.mark.fast
def test_debug_buffer_plugin_is_explicitly_enabled() -> None:
    observable_logger = ObservableLogger.with_debug_buffer(
        config=ObservabilityConfig(enabled=True, buffer_size={"*": 2})
    )

    observable_logger.info("stage", "hello")

    assert observable_logger.debug_buffer is not None
    records = observable_logger.debug_buffer.drain()
    assert len(records) == 1
    assert records[0]["message"].endswith("hello")
