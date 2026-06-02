from __future__ import annotations

import pytest

from gage_eval.observability.config import ObservabilityConfig


@pytest.mark.fast
def test_observability_config_from_env_covers_runtime_log_sink_and_http(monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_OBSERVABILITY", "true")
    monkeypatch.setenv("GAGE_EVAL_INMEMORY_TRACE", "1")
    monkeypatch.setenv("GAGE_EVAL_TRACE_BUFFER_MAX_EVENTS", "17")
    monkeypatch.setenv("GAGE_EVAL_ENABLE_LOG_SINK", "false")
    monkeypatch.setenv("GAGE_EVAL_LOG_SINK_LEVEL", "WARNING")
    monkeypatch.setenv("GAGE_EVAL_LOG_SINK_MAX_QUEUE", "23")
    monkeypatch.setenv("GAGE_EVAL_LOG_SINK_FLUSH_INTERVAL_S", "0.75")
    monkeypatch.setenv("GAGE_EVAL_LOG_SINK_BATCH_SIZE", "11")
    monkeypatch.setenv("GAGE_EVAL_LOG_SINK_ZOMBIE_ROUTE_TTL_S", "9.5")
    monkeypatch.setenv("GAGE_EVAL_LOG_SINK_ROUTE_SWEEP_INTERVAL_S", "4.5")
    monkeypatch.setenv("GAGE_EVAL_LOG_SINK_CLOSE_MODE", "best_effort")
    monkeypatch.setenv("GAGE_EVAL_LOG_SINK_DRAIN_TIMEOUT_S", "1.25")
    monkeypatch.setenv("GAGE_EVAL_REPORT_HTTP_URL", "https://collector.example/events")
    monkeypatch.setenv("GAGE_EVAL_REPORT_HTTP_BATCH", "13")
    monkeypatch.setenv("GAGE_EVAL_REPORT_HTTP_FAIL_PCT", "7.5")
    monkeypatch.setenv("GAGE_EVAL_REPORT_HTTP_TIMEOUT", "3.25")
    monkeypatch.setenv("GAGE_EVAL_REPORT_HTTP_MAX_RETRIES", "5")
    monkeypatch.setenv("GAGE_EVAL_REPORT_HTTP_RETRY_BASE_MS", "25")
    monkeypatch.setenv("GAGE_EVAL_REPORT_HTTP_RETRY_MAX_MS", "250")
    monkeypatch.setenv("GAGE_EVAL_REPORT_HTTP_RETRY_MULTIPLIER", "1.5")

    config = ObservabilityConfig.from_env()

    assert config.enabled is True
    assert config.inmemory_trace is True
    assert config.trace_buffer_max_events == 17
    assert config.log_sink_enabled is False
    assert config.log_sink_level == "WARNING"
    assert config.log_sink_max_queue == 23
    assert config.log_sink_flush_interval_s == 0.75
    assert config.log_sink_batch_size == 11
    assert config.log_sink_zombie_route_ttl_s == 9.5
    assert config.log_sink_route_sweep_interval_s == 4.5
    assert config.log_sink_close_mode == "best_effort"
    assert config.log_sink_drain_timeout_s == 1.25
    assert config.report_http_url == "https://collector.example/events"
    assert config.report_http_batch == 13
    assert config.report_http_fail_pct == 7.5
    assert config.report_http_timeout == 3.25
    assert config.report_http_max_retries == 5
    assert config.report_http_retry_base_ms == 25.0
    assert config.report_http_retry_max_ms == 250.0
    assert config.report_http_retry_multiplier == 1.5


@pytest.mark.fast
def test_observability_config_sanitizes_invalid_env_values(monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_TRACE_BUFFER_MAX_EVENTS", "bad")
    monkeypatch.setenv("GAGE_EVAL_LOG_SINK_MAX_QUEUE", "-10")
    monkeypatch.setenv("GAGE_EVAL_LOG_SINK_BATCH_SIZE", "0")
    monkeypatch.setenv("GAGE_EVAL_LOG_SINK_FLUSH_INTERVAL_S", "-1")
    monkeypatch.setenv("GAGE_EVAL_LOG_SINK_CLOSE_MODE", "invalid")
    monkeypatch.setenv("GAGE_EVAL_REPORT_HTTP_MAX_RETRIES", "-3")
    monkeypatch.setenv("GAGE_EVAL_REPORT_HTTP_RETRY_MULTIPLIER", "0.5")

    config = ObservabilityConfig.from_env()

    assert config.trace_buffer_max_events == 2048
    assert config.log_sink_max_queue == 1
    assert config.log_sink_batch_size == 1
    assert config.log_sink_flush_interval_s == 0.05
    assert config.log_sink_close_mode == "drain"
    assert config.report_http_max_retries == 0
    assert config.report_http_retry_multiplier == 1.0
