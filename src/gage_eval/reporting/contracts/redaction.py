from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SecretPattern:
    """Describes a secret detection pattern."""

    kind: str
    name: str
    regex: str

    @property
    def placeholder(self) -> str:
        """Returns the canonical redaction placeholder."""
        return f"<redacted:{self.kind}>"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecretPattern":
        """Builds a pattern from a JSON-compatible mapping."""
        return cls(kind=str(data["kind"]), name=str(data["name"]), regex=str(data["regex"]))

    def to_dict(self) -> dict[str, str]:
        """Serializes the pattern."""
        return {"kind": self.kind, "name": self.name, "regex": self.regex}


@dataclass(frozen=True)
class RedactionFinding:
    """Reports a secret match without storing the matched secret value."""

    kind: str
    start: int
    end: int
    pattern_name: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RedactionFinding":
        """Builds a finding from a JSON-compatible mapping."""
        return cls(
            kind=str(data["kind"]),
            start=int(data["start"]),
            end=int(data["end"]),
            pattern_name=str(data["pattern_name"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the finding without secret text."""
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "pattern_name": self.pattern_name,
        }


@dataclass
class RedactionResult:
    """Contains a redacted value and metadata about redaction."""

    value: Any
    findings: list[RedactionFinding] = field(default_factory=list)
    redacted: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RedactionResult":
        """Builds a redaction result from a JSON-compatible mapping."""
        return cls(
            value=data.get("value"),
            findings=[
                RedactionFinding.from_dict(item) for item in data.get("findings", [])
            ],
            redacted=bool(data.get("redacted", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the redaction result."""
        return {
            "value": self.value,
            "findings": [finding.to_dict() for finding in self.findings],
            "redacted": self.redacted,
        }
