from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from gage_eval.agent_runtime.serialization import to_json_compatible


@dataclass(frozen=True)
class ResourceLease:
    """Represents one acquired runtime resource lease."""

    lease_id: str
    resource_kind: Literal["docker", "local_process"]
    profile_id: str
    lifecycle: Literal["per_sample", "per_task"]
    endpoints: dict[str, str] = field(default_factory=dict)
    handle_ref: dict[str, Any] = field(default_factory=dict)
    cleanup_policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""

        return {
            "lease_id": self.lease_id,
            "resource_kind": self.resource_kind,
            "profile_id": self.profile_id,
            "lifecycle": self.lifecycle,
            "endpoints": to_json_compatible(self.endpoints or {}),
            "handle_ref": to_json_compatible(self.handle_ref or {}),
            "cleanup_policy": to_json_compatible(self.cleanup_policy or {}),
            "metadata": to_json_compatible(self.metadata or {}),
        }
