"""Client run request/result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ClientRunRequest:
    """Request object for running an installed client."""

    instruction: str
    cwd: str
    env: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClientRunResult:
    """Result object returned by an installed client driver."""

    exit_code: int
    stdout: str
    stderr: str
    patch_path: Optional[str] = None
    patch_content: Optional[str] = None
    trajectory_path: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
