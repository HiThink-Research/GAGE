"""ClientSurface — declares available terminal, fs, and ACP capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

SurfaceType = Literal["terminal", "fs", "acp", "browser", "desktop"]
SurfaceStatus = Literal["available", "partial", "unavailable"]


@dataclass(frozen=True)
class ClientSurface:
    """A concrete client interaction surface exposed to a runtime."""

    surface_type: SurfaceType
    status: SurfaceStatus = "available"
    capabilities: tuple[str, ...] = ()
    endpoint: Optional[str] = None
    reason: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
