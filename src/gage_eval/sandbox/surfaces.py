"""Formal client surface definitions for sandbox-backed runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

from gage_eval.sandbox.contracts import RemoteSandboxContract, RemoteSandboxHandle, coerce_remote_sandbox_handle

SurfaceType = str
SurfaceStatus = str


@dataclass(frozen=True)
class ClientSurface:
    """A concrete client interaction surface exposed to a runtime."""

    surface_type: SurfaceType
    status: SurfaceStatus = "available"
    capabilities: tuple[str, ...] = ()
    endpoint: Optional[str] = None
    reason: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


def build_remote_surfaces(
    contract: RemoteSandboxContract,
    handle: RemoteSandboxHandle | Mapping[str, Any],
) -> dict[str, ClientSurface]:
    """Derive concrete runtime surfaces from the remote contract and handle."""

    normalized_handle = coerce_remote_sandbox_handle(handle)
    surfaces: dict[str, ClientSurface] = {}

    exec_url = normalized_handle.exec_url or contract.exec_endpoint
    data_endpoint = normalized_handle.data_endpoint or contract.file_endpoint
    file_read_url = normalized_handle.file_read_url or contract.file_read_url
    file_write_url = normalized_handle.file_write_url or contract.file_write_url
    env_endpoint = normalized_handle.env_endpoint or contract.env_endpoint
    api_endpoint = normalized_handle.apis_endpoint or contract.apis_endpoint
    mcp_endpoint = normalized_handle.mcp_endpoint or contract.mcp_endpoint
    workspace_root = normalized_handle.workspace_root or contract.workspace_root or contract.attach_target

    if exec_url:
        surfaces["terminal"] = ClientSurface(
            surface_type="terminal",
            status="available",
            endpoint=exec_url,
            capabilities=("exec",),
            params={"mode": contract.mode},
        )

    if file_read_url or file_write_url or data_endpoint or exec_url:
        fs_status = "available" if (file_read_url or file_write_url or data_endpoint) else "partial"
        fs_reason = None if fs_status == "available" else "using_terminal_fallback"
        surfaces["fs"] = ClientSurface(
            surface_type="fs",
            status=fs_status,
            endpoint=data_endpoint or file_read_url or file_write_url or exec_url,
            capabilities=("read_file", "write_file"),
            reason=fs_reason,
            params={
                "file_read_url": file_read_url,
                "file_write_url": file_write_url,
                "workspace_root": workspace_root,
            },
        )

    if mcp_endpoint:
        surfaces["mcp"] = ClientSurface(
            surface_type="mcp",
            status="available",
            endpoint=mcp_endpoint,
            capabilities=("mcp",),
        )
    if api_endpoint:
        surfaces["api"] = ClientSurface(
            surface_type="api",
            status="available",
            endpoint=api_endpoint,
            capabilities=("http",),
        )
    if env_endpoint:
        surfaces["env"] = ClientSurface(
            surface_type="env",
            status="available",
            endpoint=env_endpoint,
            capabilities=("env",),
        )
    return surfaces


def serialize_surfaces(surfaces: Mapping[str, ClientSurface]) -> dict[str, dict[str, Any]]:
    """Serialize surface definitions for result payloads and verifier inputs."""

    return {str(name): asdict(surface) for name, surface in surfaces.items()}
