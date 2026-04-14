from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gage_eval.agent_runtime.session import AgentRuntimeSession


@runtime_checkable
class ClientSurface(Protocol):
    """Defines the standardized installed-client execution contract."""

    def setup(
        self,
        environment: dict[str, Any],
        session: AgentRuntimeSession,
    ) -> dict[str, Any] | None: ...

    def run(
        self,
        request: dict[str, Any],
        environment: dict[str, Any],
    ) -> dict[str, Any]: ...

    async def arun(
        self,
        request: dict[str, Any],
        environment: dict[str, Any],
    ) -> dict[str, Any]: ...
