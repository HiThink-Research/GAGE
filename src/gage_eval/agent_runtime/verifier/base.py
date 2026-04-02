"""Verifier protocol and data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from gage_eval.sandbox.surfaces import ClientSurface


@dataclass(frozen=True)
class VerifierInput:
    """Input payload for a benchmark verifier."""

    benchmark_kit_id: str
    sample_id: str
    payload: Dict[str, Any]
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    runtime_handle: Dict[str, Any] = field(default_factory=dict)
    surfaces: Dict[str, ClientSurface] = field(default_factory=dict)
    workspace_root: Optional[str] = None


@dataclass(frozen=True)
class VerifierResult:
    """Structured result returned by a verifier."""

    status: str
    score: Optional[float] = None
    summary: Optional[str] = None
    raw_output: Dict[str, Any] = field(default_factory=dict)


class Verifier(Protocol):
    """Verifier protocol."""

    def verify(self, verifier_input: VerifierInput) -> VerifierResult: ...
