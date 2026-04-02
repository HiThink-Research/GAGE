"""ResourceBundle — runtime-consumed resource container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from gage_eval.sandbox.contracts import RemoteSandboxContract
from gage_eval.sandbox.surfaces import ClientSurface


@dataclass
class ResourceBundle:
    """Container for environment, surface, and remote sandbox resources."""

    environment: Any
    remote_sandbox: Optional[RemoteSandboxContract] = None
    surfaces: Dict[str, ClientSurface] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_surface(self, name: str) -> Optional[ClientSurface]:
        return self.surfaces.get(name)

    def require_surface(self, name: str) -> ClientSurface:
        surface = self.surfaces.get(name)
        if surface is None:
            raise KeyError(f"Required surface '{name}' not available")
        return surface
