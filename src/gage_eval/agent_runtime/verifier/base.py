"""Verifier protocol and data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class VerifierInput:
    """Normalized input payload for verification."""

    benchmark_kit_id: str
    sample_id: str
    payload: Dict[str, Any]
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierResult:
    """Normalized verifier output."""

    status: str
    score: Optional[float] = None
    summary: Optional[str] = None
    raw_output: Dict[str, Any] = field(default_factory=dict)


class Verifier(Protocol):
    """Sync verification protocol."""

    def verify(self, verifier_input: VerifierInput) -> VerifierResult:
        """Verify a scheduler result."""
