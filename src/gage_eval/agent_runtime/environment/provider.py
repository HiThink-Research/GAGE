"""Environment construction helpers."""

from __future__ import annotations

from gage_eval.agent_runtime.environment.base import AgentEnvironment


class EnvironmentProvider:
    """Phase 1 main-path factory."""

    def build(self, plan, sample) -> AgentEnvironment:
        raise NotImplementedError("EnvironmentProvider.build is not implemented yet")
