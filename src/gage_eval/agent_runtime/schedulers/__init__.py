"""Scheduler protocol and result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from gage_eval.agent_runtime.session import AgentRuntimeSession


@dataclass
class SchedulerResult:
    """Normalized scheduler output."""

    status: str
    answer: Optional[str] = None
    patch_path: Optional[str] = None
    stdout_path: Optional[str] = None
    trajectory_path: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw_output: Dict[str, Any] = field(default_factory=dict)


class Scheduler(Protocol):
    """Common sync scheduler protocol for phase 1 runtimes."""

    def run(self, session: AgentRuntimeSession) -> SchedulerResult:
        """Run a compiled runtime session."""


__all__ = ["Scheduler", "SchedulerResult"]
