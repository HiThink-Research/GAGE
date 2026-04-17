from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.agent_runtime.verifier.contracts import RuntimeJudgeOutcome


@dataclass
class AgentRuntimeSession:
    """Tracks one sample-scoped runtime transaction."""

    session_id: str
    run_id: str
    task_id: str
    sample_id: str
    benchmark_kit_id: str
    scheduler_type: str
    client_id: str | None = None
    resource_lease: ResourceLease | None = None
    runtime_context: dict[str, Any] = field(default_factory=dict)
    prompt_context: dict[str, Any] = field(default_factory=dict)
    benchmark_state: dict[str, Any] = field(default_factory=dict)
    scheduler_state: dict[str, Any] = field(default_factory=dict)
    artifact_layout: dict[str, str] = field(default_factory=dict)
    judge_outcome: RuntimeJudgeOutcome | None = None
