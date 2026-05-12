from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class HostBridgeInfo:
    network_mode: Optional[str]
    host_alias: str
    extra_hosts: list[str]


def probe_host_bridge(runtime_configs: Dict[str, Any]) -> HostBridgeInfo:
    """Resolve host gateway hints from Docker runtime configs.

    Args:
        runtime_configs: Raw runtime_configs mapping from a sandbox profile.

    Returns:
        HostBridgeInfo with normalized network mode, host alias, and extra_hosts.
    """

    normalized = normalize_runtime_configs(runtime_configs or {})
    network_mode = normalized.get("network_mode")
    extra_hosts = list(normalized.get("extra_hosts") or [])
    host_alias = _resolve_host_alias(network_mode, extra_hosts)
    return HostBridgeInfo(network_mode=network_mode, host_alias=host_alias, extra_hosts=extra_hosts)


def _resolve_host_alias(network_mode: Optional[str], extra_hosts: list[str]) -> str:
    if network_mode == "host":
        return "127.0.0.1"
    for entry in extra_hosts:
        if str(entry).startswith("host.docker.internal:"):
            return "host.docker.internal"
    return "127.0.0.1"


def normalize_runtime_configs(runtime_configs: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(runtime_configs or {})
    network_mode = normalized.get("network_mode")
    if network_mode in {"bridge_host", "host_gateway"}:
        normalized["network_mode"] = "bridge"
        extra_hosts = list(normalized.get("extra_hosts") or [])
        host_gateway = "host.docker.internal:host-gateway"
        if host_gateway not in extra_hosts:
            extra_hosts.append(host_gateway)
        normalized["extra_hosts"] = extra_hosts
    elif network_mode in {"host", "none"}:
        normalized["network_mode"] = network_mode
    return normalized
