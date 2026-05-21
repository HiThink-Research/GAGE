from __future__ import annotations

from typing import Any


Diagnostic = dict[str, Any]


class SeverityValue(str):
    """String severity value with enum-like `.value` compatibility."""

    @property
    def value(self) -> str:
        return str(self)


class Severity:
    """Defines the report-wide severity vocabulary."""

    CRITICAL = SeverityValue("critical")
    HIGH = SeverityValue("high")
    MEDIUM = SeverityValue("medium")
    LOW = SeverityValue("low")
    INFO = SeverityValue("info")

    VALUES = (CRITICAL, HIGH, MEDIUM, LOW, INFO)
    _RANK = {CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1, INFO: 0}

    @classmethod
    def validate(cls, value: object, path: str = "severity") -> list[Diagnostic]:
        """Returns diagnostics for unsupported severity values."""
        if value in cls.VALUES:
            return []
        return [
            {
                "code": "report_context.invalid_enum",
                "path": path,
                "message": f"Unsupported severity: {value}",
            }
        ]

    @classmethod
    def rank(cls, value: str) -> int:
        """Returns a sortable severity rank."""
        return cls._RANK.get(value, -1)

    @classmethod
    def parse(cls, value: object) -> SeverityValue:
        text = SeverityValue(str(value))
        if text not in cls.VALUES:
            raise ValueError(f"Unsupported severity: {value}")
        return text
