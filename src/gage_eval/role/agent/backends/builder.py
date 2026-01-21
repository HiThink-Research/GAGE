"""Builder utilities for agent backends."""

from __future__ import annotations

from typing import Any, Dict, Type

from gage_eval.role.agent.backends.base import AgentBackend
from gage_eval.role.agent.backends.class_backend import ClassBackend
from gage_eval.role.agent.backends.cli_backend import CliBackend
from gage_eval.role.agent.backends.http_backend import HttpBackend
from gage_eval.role.agent.backends.mcp_backend import McpBackend
from gage_eval.role.agent.backends.model_backend import ModelBackend


_BACKEND_TYPES: Dict[str, Type[AgentBackend]] = {
    "agent_class": ClassBackend,
    "agent_cli": CliBackend,
    "agent_http": HttpBackend,
    "agent_mcp": McpBackend,
    "model_backend": ModelBackend,
    "class": ClassBackend,
    "cli": CliBackend,
    "http": HttpBackend,
    "mcp": McpBackend,
}


def build_agent_backend(spec: Dict[str, Any]) -> AgentBackend:
    backend_type = spec.get("type")
    if not backend_type:
        raise ValueError("Agent backend spec missing 'type'")
    backend_cls = _BACKEND_TYPES.get(backend_type)
    if backend_cls is None:
        raise KeyError(f"Unsupported agent backend type '{backend_type}'")
    config = spec.get("config") or {}
    return backend_cls(config)
