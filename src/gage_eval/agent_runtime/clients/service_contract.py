from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from gage_eval.agent_runtime.clients.types import ClientRunRequest


@dataclass(frozen=True)
class InstalledClientServiceRequest:
    """Represents the HTTP request payload sent to one installed-client service."""

    request: ClientRunRequest
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the service request to one JSON-friendly payload."""

        return {
            "request": self.request.to_dict(),
            "environment": dict(self.environment or {}),
        }


@dataclass(frozen=True)
class InstalledClientServiceResult:
    """Represents the normalized response returned by one installed-client service."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    answer: str = ""
    status: str = "completed"
    patch_content: str | None = None
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    trajectory_text: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "InstalledClientServiceResult":
        """Builds the normalized result from a service JSON payload."""

        normalized = dict(payload or {})
        agent_trace = normalized.get("agent_trace")
        metadata = normalized.get("metadata")
        return cls(
            exit_code=_coerce_exit_code(normalized.get("exit_code")),
            stdout=str(normalized.get("stdout") or ""),
            stderr=str(normalized.get("stderr") or ""),
            answer=str(normalized.get("answer") or ""),
            status=str(normalized.get("status") or "completed"),
            patch_content=_coerce_optional_string(normalized.get("patch_content")),
            agent_trace=_normalize_agent_trace(agent_trace),
            metadata=dict(metadata or {}) if isinstance(metadata, Mapping) else {},
            trajectory_text=_coerce_optional_string(normalized.get("trajectory_text"))
            or _coerce_optional_string(normalized.get("trajectory")),
            usage=dict(normalized.get("usage") or {})
            if isinstance(normalized.get("usage"), Mapping)
            else {},
        )


def _normalize_agent_trace(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            normalized.append({str(key): entry for key, entry in item.items()})
    return normalized


def _coerce_optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _coerce_exit_code(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
