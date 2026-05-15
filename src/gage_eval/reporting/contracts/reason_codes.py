from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml


@dataclass(frozen=True)
class ReasonCodeEntry:
    """Stores display and default scoring metadata for a reason code."""

    code: str
    impact_default: str
    actionability_default: str
    human_readable_zh: str
    human_readable_en: str

    @classmethod
    def from_dict(cls, code: str, data: dict[str, Any]) -> "ReasonCodeEntry":
        """Builds a reason code entry from a registry mapping."""
        return cls(
            code=code,
            impact_default=str(data["impact_default"]),
            actionability_default=str(data["actionability_default"]),
            human_readable_zh=str(data["human_readable_zh"]),
            human_readable_en=str(data["human_readable_en"]),
        )

    def to_dict(self) -> dict[str, str]:
        """Serializes a reason code entry."""
        return {
            "impact_default": self.impact_default,
            "actionability_default": self.actionability_default,
            "human_readable_zh": self.human_readable_zh,
            "human_readable_en": self.human_readable_en,
        }

    def __getitem__(self, key: str) -> str:
        return self.to_dict()[key]


@dataclass
class ReasonCodeRegistry:
    """Loads and validates registered report reason codes."""

    schema_version: str
    reason_codes: dict[str, ReasonCodeEntry]

    @property
    def codes(self) -> tuple[str, ...]:
        """Returns registered codes in sorted order."""
        return tuple(sorted(self.reason_codes))

    @classmethod
    def load_builtin(cls) -> "ReasonCodeRegistry":
        """Loads the built-in reason code registry from package resources."""
        registry_text = (
            resources.files("gage_eval.reporting.contracts")
            .joinpath("reason_codes.yaml")
            .read_text(encoding="utf-8")
        )
        return cls.from_dict(yaml.safe_load(registry_text) or {})

    @classmethod
    def load_default(cls) -> "ReasonCodeRegistry":
        """Compatibility alias for the built-in reason code registry."""
        return cls.load_builtin()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReasonCodeRegistry":
        """Builds a registry from a JSON-compatible mapping."""
        entries = {
            code: ReasonCodeEntry.from_dict(code, entry)
            for code, entry in (data.get("reason_codes") or {}).items()
        }
        return cls(
            schema_version=str(data.get("schema_version", "gage.reason_codes.v1")),
            reason_codes=entries,
        )

    def get(self, code: str) -> ReasonCodeEntry:
        """Returns a registered reason code entry."""
        return self.reason_codes[code]

    def to_dict(self) -> dict[str, Any]:
        """Serializes the registry."""
        return {
            "schema_version": self.schema_version,
            "reason_codes": {
                code: entry.to_dict() for code, entry in sorted(self.reason_codes.items())
            },
        }

    def validate_completeness(self, declared_codes: list[str] | tuple[str, ...] | set[str]) -> list[dict[str, Any]]:
        """Returns diagnostics for declared codes absent from the registry."""
        diagnostics: ReasonCodeDiagnostics = ReasonCodeDiagnostics()
        for code in sorted(set(declared_codes)):
            if code not in self.reason_codes:
                diagnostics.append(
                    {
                        "code": "reason_code.unregistered",
                        "path": f"reason_codes.{code}",
                        "message": f"Reason code is declared but not registered: {code}",
                    }
                )
        return diagnostics


class ReasonCodeDiagnostics(list[dict[str, Any]]):
    """Diagnostics list with compatibility equality for older missing-code tests."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, list) and all(isinstance(item, str) for item in other):
            missing = [
                str(item.get("path", "")).removeprefix("reason_codes.")
                for item in self
                if isinstance(item, dict)
            ]
            return missing == other
        return super().__eq__(other)
