"""Runtime configuration helpers for observability features."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional, Set


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _sanitize_float(value: Any, *, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _normalize_float_map(values: Optional[Dict[str, Any]]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    if not values:
        return normalized
    for key, raw in values.items():
        normalized[str(key)] = _sanitize_float(raw, default=1.0)
    return normalized


def _normalize_int_map(values: Optional[Dict[str, Any]]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
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
    sample_rate: Dict[str, float] = field(default_factory=dict)
    buffer_size: Dict[str, int] = field(default_factory=dict)
    log_first_n: Dict[str, int] = field(default_factory=dict)
    force_sample_ids: Set[str] = field(default_factory=set)
    _timings: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @classmethod
    def from_payload(cls, payload: Optional[Dict[str, Any]] = None) -> "ObservabilityConfig":
        """Create a config instance from raw dict + env fallback."""

        data = payload or {}
        env_enabled = _env_flag("GAGE_EVAL_OBSERVABILITY", default=False)
        enabled = bool(data.get("enabled", env_enabled))
        default_sample_rate = _sanitize_float(data.get("default_sample_rate"), default=1.0)
        sample_rate = _normalize_float_map(data.get("sample_rate"))
        buffer_size = _normalize_int_map(data.get("buffer_size"))
        log_first_n = _normalize_int_map(data.get("log_first_n"))
        force_samples = {str(item) for item in data.get("force_samples", []) if item is not None}
        return cls(
            enabled=enabled,
            default_sample_rate=default_sample_rate,
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            log_first_n=log_first_n,
            force_sample_ids=force_samples,
        )

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

    def snapshot_timings(self) -> Dict[str, Dict[str, float]]:
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
