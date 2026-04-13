"""Shared context passed through support units."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SupportContext:
    payload: dict[str, object] = field(default_factory=dict)
    state: dict[str, object] = field(default_factory=dict)
    unit_trace: list[str] = field(default_factory=list)
