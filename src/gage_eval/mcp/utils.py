"""Utilities for MCP client lifecycle handling."""

from __future__ import annotations

from typing import Any, Dict, Optional


def sync_mcp_endpoint(mcp_client: Any, runtime_handle: Dict[str, Any]) -> None:
    """Sync MCP client endpoint to the runtime handle if available."""

    endpoint = _extract_mcp_endpoint(runtime_handle)
    if not endpoint:
        return
    current = getattr(mcp_client, "endpoint", None)
    if current == endpoint:
        return
    disconnect = getattr(mcp_client, "disconnect", None)
    if callable(disconnect):
        disconnect()
    try:
        setattr(mcp_client, "endpoint", endpoint)
    except Exception:
        return


def _extract_mcp_endpoint(runtime_handle: Dict[str, Any]) -> Optional[str]:
    for key in ("mcp_endpoint", "mcp_url", "remote_mcp_url"):
        value = runtime_handle.get(key)
        if value:
            return str(value).rstrip("/")
    return None


__all__ = ["sync_mcp_endpoint"]
