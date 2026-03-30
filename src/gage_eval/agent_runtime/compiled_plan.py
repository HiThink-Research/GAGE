"""Compiled runtime plan — cold-path product, reused across samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from gage_eval.agent_runtime.spec import AgentRuntimeSpec, SchedulerType


@dataclass(frozen=True)
class CompiledRuntimePlan:
    """Normalized plan consumed by environment and scheduler layers."""

    runtime_spec: AgentRuntimeSpec
    scheduler_type: SchedulerType
    benchmark_kit_id: str
    client_id: Optional[str]
    role_adapter_id: Optional[str]
    environment_kind: Literal["docker", "remote", "fake"]
    required_surfaces: tuple[str, ...] = ()
    optional_surfaces: tuple[str, ...] = ()
    sandbox_profile_id: Optional[str] = None
    remote_mode: Optional[Literal["attached", "managed"]] = None
    params: Dict[str, Any] = field(default_factory=dict)
