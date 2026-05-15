from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any


Diagnostic = dict[str, Any]


@dataclass
class EvidenceRef:
    """References report evidence by run-directory relative path."""

    ref_id: str | None = None
    kind: str | None = None
    path: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    timestamp_iso: str | None = None
    scenario_kind: str | None = None
    artifact_role: str | None = None
    sample_id: str | None = None
    task_id: str | None = None
    trial_id: str | None = None
    preview: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvidenceRef":
        """Builds an evidence reference from a JSON-compatible mapping."""
        payload = dict(data or {})
        known = {field_name: payload.pop(field_name, None) for field_name in _FIELDS}
        return cls(extra=payload, **known)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the evidence reference."""
        data = {field_name: getattr(self, field_name) for field_name in _FIELDS}
        data = {key: value for key, value in data.items() if value is not None}
        data.update(self.extra)
        return data

    def validate(self, path: str = "evidence_refs[]") -> list[Diagnostic]:
        """Returns diagnostics for invalid evidence references."""
        diagnostics: list[Diagnostic] = []
        for field_name in ("ref_id", "kind", "path", "mime_type", "size_bytes", "sha256", "timestamp_iso"):
            if getattr(self, field_name) is None:
                diagnostics.append(
                    {
                        "code": "report_context.required_missing",
                        "path": f"{path}.{field_name}",
                    }
                )

        if self.path and _is_absolute_or_escaping(self.path):
            diagnostics.append(
                {
                    "code": "report_context.path_not_relative",
                    "path": f"{path}.path",
                    "message": "EvidenceRef.path must be run_dir relative",
                }
            )
        return diagnostics


_FIELDS = (
    "ref_id",
    "kind",
    "path",
    "scenario_kind",
    "artifact_role",
    "sample_id",
    "task_id",
    "trial_id",
    "mime_type",
    "size_bytes",
    "sha256",
    "timestamp_iso",
    "preview",
)


def _is_absolute_or_escaping(value: str) -> bool:
    if value.startswith(("/", "~")):
        return True
    path = PurePosixPath(value)
    return any(part == ".." for part in path.parts)
