from __future__ import annotations

from gage_eval.agent_runtime.clients.builder import (
    LegacyInvokeClientSurface,
    StructuredClientSurfaceAdapter,
    build_client_surface,
    instantiate_builtin_client,
    resolve_installed_client,
)
from gage_eval.agent_runtime.clients.codex import CodexClient
from gage_eval.agent_runtime.clients.contracts import ClientSurface
from gage_eval.agent_runtime.clients.runner import InstalledClientRunner
from gage_eval.agent_runtime.clients.types import ClientRunRequest, ClientRunResult

__all__ = [
    "build_client_surface",
    "ClientSurface",
    "ClientRunRequest",
    "ClientRunResult",
    "CodexClient",
    "InstalledClientRunner",
    "instantiate_builtin_client",
    "LegacyInvokeClientSurface",
    "resolve_installed_client",
    "StructuredClientSurfaceAdapter",
]
