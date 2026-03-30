"""Agent runtime specification — declares scheduler, kit, and resource policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

SchedulerType = Literal["framework_loop", "installed_client", "acp_client"]
EnvironmentKind = Literal["docker", "remote", "fake"]
RemoteSandboxMode = Literal["attached", "managed"]


@dataclass(frozen=True)
class ResourcePolicy:
    """Resource requirements for agent execution."""

    environment_kind: EnvironmentKind = "docker"
    mcp_client_ids: tuple[str, ...] = ()
    timeout_sec: int = 1800
    env: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClientSurfacePolicy:
    """Declares which client surfaces are required or optional."""

    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    fail_on_missing_required: bool = True


@dataclass(frozen=True)
class SandboxPolicy:
    """Sandbox/environment allocation policy."""

    sandbox_profile_id: Optional[str] = None
    prefer_remote: bool = False
    allow_local_fallback: bool = True
    remote_mode: Optional[RemoteSandboxMode] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentRuntimeSpec:
    """Top-level runtime spec, one per agent_runtime_id."""

    agent_runtime_id: str
    scheduler: SchedulerType
    benchmark_kit_id: str
    client_id: Optional[str] = None
    role_adapter_id: Optional[str] = None
    resource_policy: ResourcePolicy = field(default_factory=ResourcePolicy)
    client_surface_policy: ClientSurfacePolicy = field(default_factory=ClientSurfacePolicy)
    sandbox_policy: SandboxPolicy = field(default_factory=SandboxPolicy)
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_runtime_id:
            raise ValueError("agent_runtime_id is required")
