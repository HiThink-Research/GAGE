from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gage_eval.agent_runtime.contracts.failure import FailureEnvelope
from gage_eval.agent_runtime.serialization import to_json_compatible


@dataclass(frozen=True)
class VerifierInput:
    """Captures the normalized verifier request."""

    benchmark_kit_id: str
    scheduler_type: str
    sample_id: str
    sample: dict[str, Any]
    scheduler_result: dict[str, Any]
    runtime_context: dict[str, Any] = field(default_factory=dict)
    verifier_resources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""

        return {
            "benchmark_kit_id": self.benchmark_kit_id,
            "scheduler_type": self.scheduler_type,
            "sample_id": self.sample_id,
            "sample": to_json_compatible(self.sample or {}),
            "scheduler_result": to_json_compatible(self.scheduler_result or {}),
            "runtime_context": to_json_compatible(self.runtime_context or {}),
            "verifier_resources": to_json_compatible(self.verifier_resources or {}),
        }


@dataclass(frozen=True)
class VerifierResult:
    """Captures the raw verifier result before judge normalization."""

    status: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""

        return {
            "status": self.status,
            "payload": to_json_compatible(self.payload or {}),
        }


@dataclass
class RuntimeJudgeOutcome:
    """Captures the runtime-owned judge result and persistence metadata."""

    verifier_input: VerifierInput
    verifier_result: VerifierResult
    judge_output: dict[str, Any]
    persisted_path: str
    failure: FailureEnvelope | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""

        return {
            "verifier_input": self.verifier_input.to_dict(),
            "verifier_result": self.verifier_result.to_dict(),
            "judge_output": to_json_compatible(self.judge_output or {}),
            "persisted_path": self.persisted_path,
            "failure": self.failure.to_dict() if self.failure is not None else None,
        }
