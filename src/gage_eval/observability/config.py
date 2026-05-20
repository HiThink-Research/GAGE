"""Runtime configuration helpers for observability features."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Mapping, Optional


OBSERVABILITY_ENV_NAMES = frozenset(
    {
        "GAGE_EVAL_OBSERVABILITY",
        "GAGE_EVAL_INMEMORY_TRACE",
        "GAGE_EVAL_TRACE_BUFFER_MAX_EVENTS",
        "GAGE_EVAL_ENABLE_LOG_SINK",
        "GAGE_EVAL_LOG_SINK_LEVEL",
        "GAGE_EVAL_LOG_SINK_MAX_QUEUE",
        "GAGE_EVAL_LOG_SINK_FLUSH_INTERVAL_S",
        "GAGE_EVAL_LOG_SINK_BATCH_SIZE",
        "GAGE_EVAL_LOG_SINK_ZOMBIE_ROUTE_TTL_S",
        "GAGE_EVAL_LOG_SINK_ROUTE_SWEEP_INTERVAL_S",
        "GAGE_EVAL_LOG_SINK_CLOSE_MODE",
        "GAGE_EVAL_LOG_SINK_DRAIN_TIMEOUT_S",
        "GAGE_EVAL_REPORT_HTTP_URL",
        "GAGE_EVAL_REPORT_HTTP_BATCH",
        "GAGE_EVAL_REPORT_HTTP_FAIL_PCT",
        "GAGE_EVAL_REPORT_HTTP_TIMEOUT",
        "GAGE_EVAL_REPORT_HTTP_MAX_RETRIES",
        "GAGE_EVAL_REPORT_HTTP_RETRY_BASE_MS",
        "GAGE_EVAL_REPORT_HTTP_RETRY_MAX_MS",
        "GAGE_EVAL_REPORT_HTTP_RETRY_MULTIPLIER",
    }
)


def _env_flag(env: Mapping[str, str], name: str, *, default: bool = False) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _sanitize_float(
    value: Any,
    *,
    default: float,
    minimum: float = 0.0,
    maximum: float | None = 1.0,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if maximum is None:
        return max(minimum, parsed)
    return max(minimum, min(maximum, parsed))


def _sanitize_int(value: Any, *, default: int, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _normalize_float_map(values: Optional[dict[str, Any]]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    if not values:
        return normalized
    for key, raw in values.items():
        normalized[str(key)] = _sanitize_float(raw, default=1.0)
    return normalized


def _normalize_int_map(values: Optional[dict[str, Any]]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    if not values:
        return normalized
    for key, raw in values.items():
        try:
            normalized[str(key)] = max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return normalized


@dataclass
class ObservabilityConfig:
    """Holds runtime flags for observability sampling/buffering."""

    enabled: bool = False
    default_sample_rate: float = 1.0
    sample_rate: dict[str, float] = field(default_factory=dict)
    buffer_size: dict[str, int] = field(default_factory=dict)
    log_first_n: dict[str, int] = field(default_factory=dict)
    force_sample_ids: set[str] = field(default_factory=set)
    inmemory_trace: bool = False
    trace_buffer_max_events: int = 2048
    log_sink_enabled: bool = True
    log_sink_level: str = "INFO"
    log_sink_max_queue: int = 1024
    log_sink_flush_interval_s: float = 0.25
    log_sink_batch_size: int = 64
    log_sink_zombie_route_ttl_s: float = 300.0
    log_sink_route_sweep_interval_s: float = 30.0
    log_sink_close_mode: str = "drain"
    log_sink_drain_timeout_s: float = 2.0
    report_http_url: Optional[str] = None
    report_http_batch: int = 50
    report_http_fail_pct: float = 5.0
    report_http_timeout: float = 10.0
    report_http_max_retries: int = 2
    report_http_retry_base_ms: float = 50.0
    report_http_retry_max_ms: float = 500.0
    report_http_retry_multiplier: float = 2.0
    _timings: dict[str, dict[str, float]] = field(default_factory=dict, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "ObservabilityConfig":
        """Create a config instance from observability environment variables."""

        values = os.environ if env is None else env
        close_mode = (values.get("GAGE_EVAL_LOG_SINK_CLOSE_MODE", "drain").strip().lower() or "drain")
        if close_mode not in {"drain", "best_effort"}:
            close_mode = "drain"
        return cls(
            enabled=_env_flag(values, "GAGE_EVAL_OBSERVABILITY", default=False),
            inmemory_trace=_env_flag(values, "GAGE_EVAL_INMEMORY_TRACE", default=False),
            trace_buffer_max_events=_sanitize_int(
                values.get("GAGE_EVAL_TRACE_BUFFER_MAX_EVENTS"),
                default=2048,
                minimum=0,
            ),
            log_sink_enabled=_env_flag(values, "GAGE_EVAL_ENABLE_LOG_SINK", default=True),
            log_sink_level=values.get("GAGE_EVAL_LOG_SINK_LEVEL", "INFO") or "INFO",
            log_sink_max_queue=_sanitize_int(values.get("GAGE_EVAL_LOG_SINK_MAX_QUEUE"), default=1024, minimum=1),
            log_sink_flush_interval_s=_sanitize_float(
                values.get("GAGE_EVAL_LOG_SINK_FLUSH_INTERVAL_S"),
                default=0.25,
                minimum=0.05,
                maximum=None,
            ),
            log_sink_batch_size=_sanitize_int(values.get("GAGE_EVAL_LOG_SINK_BATCH_SIZE"), default=64, minimum=1),
            log_sink_zombie_route_ttl_s=_sanitize_float(
                values.get("GAGE_EVAL_LOG_SINK_ZOMBIE_ROUTE_TTL_S"),
                default=300.0,
                minimum=1.0,
                maximum=None,
            ),
            log_sink_route_sweep_interval_s=_sanitize_float(
                values.get("GAGE_EVAL_LOG_SINK_ROUTE_SWEEP_INTERVAL_S"),
                default=30.0,
                minimum=1.0,
                maximum=None,
            ),
            log_sink_close_mode=close_mode,
            log_sink_drain_timeout_s=_sanitize_float(
                values.get("GAGE_EVAL_LOG_SINK_DRAIN_TIMEOUT_S"),
                default=2.0,
                minimum=0.0,
                maximum=None,
            ),
            report_http_url=values.get("GAGE_EVAL_REPORT_HTTP_URL") or None,
            report_http_batch=_sanitize_int(values.get("GAGE_EVAL_REPORT_HTTP_BATCH"), default=50, minimum=1),
            report_http_fail_pct=_sanitize_float(
                values.get("GAGE_EVAL_REPORT_HTTP_FAIL_PCT"),
                default=5.0,
                minimum=0.0,
                maximum=100.0,
            ),
            report_http_timeout=_sanitize_float(
                values.get("GAGE_EVAL_REPORT_HTTP_TIMEOUT"),
                default=10.0,
                minimum=0.0,
                maximum=None,
            ),
            report_http_max_retries=_sanitize_int(
                values.get("GAGE_EVAL_REPORT_HTTP_MAX_RETRIES"),
                default=2,
                minimum=0,
            ),
            report_http_retry_base_ms=_sanitize_float(
                values.get("GAGE_EVAL_REPORT_HTTP_RETRY_BASE_MS"),
                default=50.0,
                minimum=0.0,
                maximum=None,
            ),
            report_http_retry_max_ms=_sanitize_float(
                values.get("GAGE_EVAL_REPORT_HTTP_RETRY_MAX_MS"),
                default=500.0,
                minimum=0.0,
                maximum=None,
            ),
            report_http_retry_multiplier=_sanitize_float(
                values.get("GAGE_EVAL_REPORT_HTTP_RETRY_MULTIPLIER"),
                default=2.0,
                minimum=1.0,
                maximum=None,
            ),
        )

    @classmethod
    def from_payload(cls, payload: Optional[dict[str, Any]] = None) -> "ObservabilityConfig":
        """Create a config instance from raw dict + env fallback."""

        data = payload or {}
        env_config = cls.from_env()
        enabled = bool(data.get("enabled", env_config.enabled))
        default_sample_rate = _sanitize_float(data.get("default_sample_rate"), default=1.0)
        sample_rate = _normalize_float_map(data.get("sample_rate"))
        buffer_size = _normalize_int_map(data.get("buffer_size"))
        log_first_n = _normalize_int_map(data.get("log_first_n"))
        force_samples = {str(item) for item in data.get("force_samples", []) if item is not None}
        config = cls(
            enabled=enabled,
            default_sample_rate=default_sample_rate,
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            log_first_n=log_first_n,
            force_sample_ids=force_samples,
            inmemory_trace=env_config.inmemory_trace,
            trace_buffer_max_events=env_config.trace_buffer_max_events,
            log_sink_enabled=env_config.log_sink_enabled,
            log_sink_level=env_config.log_sink_level,
            log_sink_max_queue=env_config.log_sink_max_queue,
            log_sink_flush_interval_s=env_config.log_sink_flush_interval_s,
            log_sink_batch_size=env_config.log_sink_batch_size,
            log_sink_zombie_route_ttl_s=env_config.log_sink_zombie_route_ttl_s,
            log_sink_route_sweep_interval_s=env_config.log_sink_route_sweep_interval_s,
            log_sink_close_mode=env_config.log_sink_close_mode,
            log_sink_drain_timeout_s=env_config.log_sink_drain_timeout_s,
            report_http_url=env_config.report_http_url,
            report_http_batch=env_config.report_http_batch,
            report_http_fail_pct=env_config.report_http_fail_pct,
            report_http_timeout=env_config.report_http_timeout,
            report_http_max_retries=env_config.report_http_max_retries,
            report_http_retry_base_ms=env_config.report_http_retry_base_ms,
            report_http_retry_max_ms=env_config.report_http_retry_max_ms,
            report_http_retry_multiplier=env_config.report_http_retry_multiplier,
        )
        return config

    def buffer_size_for(self, stage: str) -> int:
        value = self.buffer_size.get(stage, self.buffer_size.get("*", 0))
        return max(0, int(value))

    def log_first_n_for(self, stage: str) -> int:
        value = self.log_first_n.get(stage, self.log_first_n.get("*", 0))
        return max(0, int(value))

    def stage_sample_rate(self, stage: str) -> float:
        value = self.sample_rate.get(stage, self.sample_rate.get("*", self.default_sample_rate))
        return _sanitize_float(value, default=self.default_sample_rate)

    def should_sample(self, stage: str, *, sample_idx: Optional[int] = None, sample_id: Optional[str] = None) -> bool:
        if not self.enabled:
            return False
        if sample_id and sample_id in self.force_sample_ids:
            return True
        first_n = self.log_first_n_for(stage)
        if first_n and sample_idx is not None and sample_idx < first_n:
            return True
        rate = self.stage_sample_rate(stage)
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        return random.random() < rate

    def force_log(self, sample_id: str) -> None:
        if not sample_id:
            return
        with self._lock:
            self.force_sample_ids.add(str(sample_id))

    def record_timing(self, stage: str, elapsed_seconds: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            metrics = self._timings.setdefault(stage, {"count": 0, "total_s": 0.0})
            metrics["count"] += 1
            metrics["total_s"] += float(elapsed_seconds)

    def snapshot_timings(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {stage: dict(values) for stage, values in self._timings.items()}


_GLOBAL_CONFIG = ObservabilityConfig.from_payload()


def get_observability_config() -> ObservabilityConfig:
    return _GLOBAL_CONFIG


def set_observability_config(config: ObservabilityConfig) -> None:
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config


def configure_observability(payload: Optional[Dict[str, Any]]) -> ObservabilityConfig:
    config = ObservabilityConfig.from_payload(payload)
    set_observability_config(config)
    return config
