"""Judge implementation base interface."""

from __future__ import annotations

from typing import Any, Dict


class JudgeImplementation:
    def invoke(self, payload: Dict[str, Any], state: Any = None) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    async def ainvoke(self, payload: Dict[str, Any], state: Any = None) -> Dict[str, Any]:  # pragma: no cover - interface
        return self.invoke(payload, state)
