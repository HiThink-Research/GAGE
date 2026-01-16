"""Agent backend implementations."""

from gage_eval.role.agent.backends.base import AgentBackend, normalize_agent_output
from gage_eval.role.agent.backends.builder import build_agent_backend
from gage_eval.role.agent.backends.class_backend import ClassBackend
from gage_eval.role.agent.backends.cli_backend import CliBackend
from gage_eval.role.agent.backends.http_backend import HttpBackend
from gage_eval.role.agent.backends.mcp_backend import McpBackend
from gage_eval.role.agent.backends.model_backend import ModelBackend

__all__ = [
    "AgentBackend",
    "normalize_agent_output",
    "build_agent_backend",
    "ClassBackend",
    "CliBackend",
    "HttpBackend",
    "McpBackend",
    "ModelBackend",
]
