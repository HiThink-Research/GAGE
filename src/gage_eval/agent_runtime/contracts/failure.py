from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from gage_eval.agent_runtime.serialization import to_json_compatible


FailureDomain = Literal[
    "compile",
    "environment",
    "input_projection",
    "client_execution",
    "artifact_capture",
    "verifier",
    "persistence",
    "upstream_dependency",
]

FailureStage = Literal[
    "compile_spec",
    "bind_workflow",
    "acquire_lease",
    "bootstrap_runtime",
    "prepare_inputs",
    "run_scheduler",
    "capture_artifacts",
    "normalize_result",
    "run_verifier",
    "persist_outputs",
    "cleanup",
]

ComponentKind = Literal[
    "resolver",
    "runtime",
    "resource_manager",
    "scheduler",
    "client",
    "tool_router",
    "verifier_adapter",
    "artifact_sink",
    "provider",
    "compat_shim",
]


@dataclass
class FailureEnvelope:
    """Captures stable runtime failure semantics for diagnostics and recovery."""

    failure_domain: FailureDomain
    failure_stage: FailureStage
    failure_code: str
    component_kind: ComponentKind
    component_id: str
    owner: str
    retryable: bool
    summary: str
    first_bad_step: str
    suspect_files: tuple[str, ...]
    reproduction_hint: str | None = None
    normalized_signals: dict[str, Any] | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    raw_error_path: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""

        return {
            "failure_domain": self.failure_domain,
            "failure_stage": self.failure_stage,
            "failure_code": self.failure_code,
            "component_kind": self.component_kind,
            "component_id": self.component_id,
            "owner": self.owner,
            "retryable": self.retryable,
            "summary": self.summary,
            "first_bad_step": self.first_bad_step,
            "suspect_files": list(self.suspect_files),
            "reproduction_hint": self.reproduction_hint,
            "normalized_signals": to_json_compatible(self.normalized_signals or {}),
            "artifact_paths": to_json_compatible(self.artifact_paths or {}),
            "raw_error_path": self.raw_error_path,
            "details": to_json_compatible(self.details or {}),
        }


class FailureEnvelopeError(RuntimeError):
    """Internal control-flow wrapper for already-normalized failures."""

    def __init__(self, failure: FailureEnvelope) -> None:
        self.failure = failure
        super().__init__(failure.summary)
