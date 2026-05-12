from __future__ import annotations

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class FakeInstalledClient(BaseInstalledAgent):
    """Minimal installed-agent class for Harbor adapter translation tests."""

    @staticmethod
    def name() -> str:
        return "fake-installed-client"

    async def install(self, environment: BaseEnvironment) -> None:
        del environment

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        del instruction, environment, context

    def populate_context_post_run(self, context: AgentContext) -> None:
        del context
