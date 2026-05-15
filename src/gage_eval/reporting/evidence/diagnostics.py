from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceDiagnostics:
    """Collects reportable evidence-layer diagnostics."""

    warnings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    derived_detail_count: int = 0

    def warn(self, code: str, **details: Any) -> None:
        self.warnings.append({"code": code, **details})

    def error(self, code: str, **details: Any) -> None:
        self.errors.append({"code": code, **details})

    def extend(self, other: "EvidenceDiagnostics") -> None:
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        self.derived_detail_count += other.derived_detail_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "derived_detail_count": self.derived_detail_count,
        }
