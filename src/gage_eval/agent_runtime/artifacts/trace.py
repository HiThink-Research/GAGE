"""TraceEvent — append-only event record."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceEvent:
    """Minimal runtime trace event."""

    name: str
    payload: dict
    level: str = "info"
