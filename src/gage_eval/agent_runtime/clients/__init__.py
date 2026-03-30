"""Client run request/result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ClientRunRequest:
    """Execution payload passed to an installed client driver."""

    instruction: str
    cwd: str
    env: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClientRunResult:
    """Normalized client execution result."""

    exit_code: int
    stdout: str
    stderr: str
    patch_path: Optional[str] = None
    trajectory_path: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)


__all__ = ["ClientRunRequest", "ClientRunResult"]
