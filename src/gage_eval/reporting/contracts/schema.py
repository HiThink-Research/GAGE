from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Diagnostic = dict[str, Any]


@dataclass
class ReportContextSchema:
    """Represents the structured report context schema version."""

    name: str | None = None
    major: int | None = None
    minor: int | None = None
    renderer_compat: str | None = None
    generated_by: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ReportContextSchema":
        """Builds a schema object from a JSON-compatible mapping."""
        payload = dict(data or {})
        known = {
            "name": payload.pop("name", None),
            "major": payload.pop("major", None),
            "minor": payload.pop("minor", None),
            "renderer_compat": payload.pop("renderer_compat", None),
            "generated_by": payload.pop("generated_by", None),
        }
        return cls(extra=payload, **known)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the schema to a JSON-compatible mapping."""
        data: dict[str, Any] = {
            "name": self.name,
            "major": self.major,
            "minor": self.minor,
            "renderer_compat": self.renderer_compat,
            "generated_by": self.generated_by,
        }
        data.update(self.extra)
        return data

    def validate(self, path: str = "schema") -> list[Diagnostic]:
        """Returns diagnostics for missing required schema fields."""
        diagnostics: list[Diagnostic] = []
        for field_name in ("name", "major", "minor", "renderer_compat", "generated_by"):
            if getattr(self, field_name) is None:
                diagnostics.append(
                    {
                        "code": "report_context.required_missing",
                        "path": f"{path}.{field_name}",
                    }
                )
        return diagnostics
