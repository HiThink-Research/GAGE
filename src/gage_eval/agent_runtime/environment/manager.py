"""EnvironmentManager shell — reserved for later pooling support."""

from __future__ import annotations

from gage_eval.agent_runtime.environment.base import AgentEnvironment


class EnvironmentManager:
    """Shell manager reserved for future environment pooling."""

    def acquire(self, plan, sample) -> AgentEnvironment:
        raise NotImplementedError("EnvironmentManager.acquire is not implemented yet")

    def release(self, environment: AgentEnvironment) -> None:
        return None
