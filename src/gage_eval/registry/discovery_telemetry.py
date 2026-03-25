"""Lightweight telemetry for staged discovery migration."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Mapping


@dataclass(frozen=True, slots=True)
class DiscoveryTelemetrySnapshot:
    """Immutable snapshot of discovery counters."""

    counters: Mapping[str, int]
    by_kind: Mapping[str, Mapping[str, int]]


class DiscoveryTelemetry:
    """Thread-safe counter store for discovery migration signals."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: Counter[str] = Counter()
        self._by_kind: dict[str, Counter[str]] = defaultdict(Counter)

    def record(self, event: str, *, kind: str | None = None) -> None:
        event_name = str(event).strip()
        if not event_name:
            return
        with self._lock:
            self._counters[event_name] += 1
            if kind:
                self._by_kind[str(kind).strip()][event_name] += 1

    def snapshot(self) -> DiscoveryTelemetrySnapshot:
        with self._lock:
            return DiscoveryTelemetrySnapshot(
                counters=dict(self._counters),
                by_kind={kind: dict(counter) for kind, counter in self._by_kind.items()},
            )

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._by_kind.clear()


telemetry = DiscoveryTelemetry()
