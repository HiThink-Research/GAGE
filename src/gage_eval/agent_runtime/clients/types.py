from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from gage_eval.agent_runtime.serialization import to_json_compatible


@dataclass(frozen=True)
class ClientRunRequest:
    """Represents one normalized installed-client request payload."""

    instruction: str = ""
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | "ClientRunRequest" | None) -> "ClientRunRequest":
        """Builds a normalized client request from one runtime payload.

        Args:
            payload: Raw runtime payload or an already normalized request.

        Returns:
            A normalized client request object.
        """

        if isinstance(payload, cls):
            return payload
        normalized = dict(payload or {})
        env = normalized.get("env")
        metadata = normalized.get("metadata")
        return cls(
            instruction=str(normalized.get("instruction") or ""),
            cwd=str(normalized.get("cwd") or ""),
            env=_normalize_string_map(env),
            metadata=dict(metadata or {}) if isinstance(metadata, Mapping) else {},
            payload=normalized,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the request to a JSON-friendly dict.

        Returns:
            A JSON-friendly request payload.
        """

        return {
            "instruction": self.instruction,
            "cwd": self.cwd,
            "env": dict(self.env or {}),
            "metadata": to_json_compatible(dict(self.metadata or {})),
            "payload": to_json_compatible(dict(self.payload or {})),
        }


@dataclass(frozen=True)
class ClientRunResult:
    """Represents one normalized installed-client result payload."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    answer: str = ""
    status: str = "completed"
    patch_path: str | None = None
    patch_content: str | None = None
    trajectory_path: str | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def artifacts(self) -> dict[str, str]:
        """Returns artifact paths using the historical field name."""

        return dict(self.artifact_paths or {})

    def to_dict(self) -> dict[str, Any]:
        """Serializes the result to the runtime scheduler payload.

        Returns:
            A JSON-friendly scheduler output payload.
        """

        return {
            "exit_code": int(self.exit_code),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "answer": self.answer or self.stdout.strip(),
            "status": self.status,
            "patch_path": self.patch_path,
            "patch_content": self.patch_content,
            "trajectory_path": self.trajectory_path,
            "artifact_paths": dict(self.artifact_paths or {}),
            "agent_trace": to_json_compatible([dict(step) for step in self.agent_trace]),
            "metadata": to_json_compatible(dict(self.metadata or {})),
            "usage": to_json_compatible(dict(self.usage or {})),
        }


def _normalize_string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if key is not None and item is not None
    }
