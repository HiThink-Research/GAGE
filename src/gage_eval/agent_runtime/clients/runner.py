from __future__ import annotations

from typing import Any

from gage_eval.agent_runtime.clients.builder import build_client_surface
from gage_eval.agent_runtime.clients.contracts import ClientSurface
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.registry.utils import ensure_async


class InstalledClientRunner:
    """Runs one installed/external client surface."""

    def __init__(self, client: Any) -> None:
        self._client: ClientSurface = build_client_surface(client)
        self._setup = ensure_async(self._client.setup)
        self._arun_client = ensure_async(self._client.arun)

    @property
    def client_surface_name(self) -> str:
        """Return the normalized client surface type name for diagnostics."""

        return self._client.__class__.__name__

    async def arun(
        self,
        *,
        request: dict[str, Any],
        environment: dict[str, Any],
        session: AgentRuntimeSession,
    ) -> dict[str, Any]:
        """Execute the bound client through the standardized setup/run flow."""

        # STEP 1: Let the client attach any per-session state to the environment.
        setup_result = await self._setup(environment, session)
        if isinstance(setup_result, dict):
            environment.update(setup_result)

        # STEP 2: Execute the client against the normalized request payload.
        result = await self._arun_client(request, environment)
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        return {"result": result}
