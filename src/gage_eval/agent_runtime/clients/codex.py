"""Codex CLI client driver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gage_eval.agent_runtime.clients import ClientRunRequest, ClientRunResult

if TYPE_CHECKING:
    from gage_eval.agent_runtime.environment.base import AgentEnvironment
    from gage_eval.agent_runtime.session import AgentRuntimeSession


class CodexClient:
    """Drives the Codex CLI inside an AgentEnvironment."""

    def setup(self, environment: "AgentEnvironment", session: "AgentRuntimeSession") -> None:
        raise NotImplementedError("CodexClient.setup is not implemented yet")

    def run(self, request: ClientRunRequest, environment: "AgentEnvironment") -> ClientRunResult:
        raise NotImplementedError("CodexClient.run is not implemented yet")

    def cleanup(self, environment: "AgentEnvironment", session: "AgentRuntimeSession") -> None:
        try:
            raise NotImplementedError("CodexClient.cleanup is not implemented yet")
        except Exception:
            return None
