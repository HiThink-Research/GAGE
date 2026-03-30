"""RemoteSandboxContract — formal resource contract for remote environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from gage_eval.agent_runtime.spec import RemoteSandboxMode


@dataclass(frozen=True)
class RemoteSandboxContract:
    """Declares a remote sandbox resource with its endpoints and mode."""

    mode: RemoteSandboxMode
    control_endpoint: Optional[str] = None
    exec_endpoint: Optional[str] = None
    file_endpoint: Optional[str] = None
    attach_target: Optional[str] = None
    renew_supported: bool = False
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode == "managed" and not self.control_endpoint:
            raise ValueError("managed mode requires control_endpoint")
        if self.mode == "attached" and not self.exec_endpoint:
            raise ValueError("attached mode requires exec_endpoint")
