from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from gage_eval.agent_runtime.contracts.failure import FailureEnvelope
from gage_eval.agent_runtime.serialization import to_json_compatible


@dataclass
class SchedulerResult:
    """Normalized scheduler output consumed by runtime-owned verifiers."""

    scheduler_type: Literal["installed_client", "framework_loop", "acp_client"]
    benchmark_kit_id: str
    status: Literal["completed", "failed", "aborted"]
    agent_output: dict[str, Any]
    artifact_paths: dict[str, str] = field(default_factory=dict)
    runtime_state: dict[str, Any] = field(default_factory=dict)
    failure: FailureEnvelope | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view."""

        return {
            "scheduler_type": self.scheduler_type,
            "benchmark_kit_id": self.benchmark_kit_id,
            "status": self.status,
            "agent_output": to_json_compatible(self.agent_output or {}),
            "artifact_paths": to_json_compatible(self.artifact_paths or {}),
            "runtime_state": to_json_compatible(self.runtime_state or {}),
            "failure": self.failure.to_dict() if self.failure is not None else None,
        }
