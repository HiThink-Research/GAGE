from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ClientSurface(Protocol):
    """Defines the installed-client execution contract."""

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]: ...
