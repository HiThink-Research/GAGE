"""Append-only trace events for agent runtime runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class TraceEvent:
    """A single trace event emitted by a runtime component."""

    name: str
    payload: Dict[str, Any]
    level: str = "info"

