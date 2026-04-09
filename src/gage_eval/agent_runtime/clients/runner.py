from __future__ import annotations

from typing import Any

from gage_eval.registry.utils import ensure_async


class InstalledClientRunner:
    """Runs one installed/external client surface."""

    def __init__(self, client: Any) -> None:
        self._client = client
        invoker = getattr(client, "ainvoke", None) or getattr(client, "invoke", None)
        if invoker is None:
            raise TypeError("installed client requires invoke/ainvoke")
        self._invoker = ensure_async(invoker)

    async def arun(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the bound client."""

        result = await self._invoker(payload)
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        return {"result": result}
