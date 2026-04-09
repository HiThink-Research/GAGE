from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class AgentRuntimeSpec:
    """Declares the public runtime identity consumed by resolver/executor."""

    agent_runtime_id: str
    benchmark_kit_id: str
    scheduler_type: Literal["installed_client", "framework_loop", "acp_client"]
    client_id: str | None = None
    role_adapter_id: str | None = None
    sandbox_profile_id: str | None = None
    resource_policy: dict[str, Any] = field(default_factory=dict)
    verifier_binding_id: str | None = None
    compat_mode: Literal["none", "legacy_backend", "legacy_support"] = "none"
    runtime_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""

        return {
            "agent_runtime_id": self.agent_runtime_id,
            "benchmark_kit_id": self.benchmark_kit_id,
            "scheduler_type": self.scheduler_type,
            "client_id": self.client_id,
            "role_adapter_id": self.role_adapter_id,
            "sandbox_profile_id": self.sandbox_profile_id,
            "resource_policy": dict(self.resource_policy or {}),
            "verifier_binding_id": self.verifier_binding_id,
            "compat_mode": self.compat_mode,
            "runtime_overrides": dict(self.runtime_overrides or {}),
            "metadata": dict(self.metadata or {}),
        }
