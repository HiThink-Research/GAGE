"""Human input adapter for arena games."""

from __future__ import annotations

from queue import Queue
from typing import Any, Dict, Optional, Sequence

from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState


@registry.asset(
    "roles",
    "human",
    desc="Human input adapter for arena games",
    tags=("role", "human"),
    role_type="human",
)
class HumanAdapter(RoleAdapter):
    """Role adapter that collects human moves from CLI or queue."""

    def __init__(
        self,
        adapter_id: str,
        *,
        source: str = "auto",
        static_moves: Optional[Sequence[str]] = None,
        action_queue: Optional[Queue[str]] = None,
        capabilities=(),
        role_type: str = "human",
        **_,
    ) -> None:
        resolved_caps = tuple(capabilities) if capabilities else ("text",)
        super().__init__(adapter_id=adapter_id, role_type=role_type, capabilities=resolved_caps)
        self._source = source
        self._static_moves = list(static_moves or [])
        self._queue = action_queue

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        prompt = payload.get("prompt") or "Your move: "
        queue = payload.get("action_queue") or self._queue
        if self._source == "static":
            move = self._static_moves.pop(0) if self._static_moves else ""
        elif self._source == "auto":
            if queue is not None:
                move = queue.get()
            else:
                move = input(str(prompt))
        elif self._source in {"queue", "visualizer"}:
            if queue is None:
                raise ValueError("HumanAdapter queue source requires action_queue")
            move = queue.get()
        else:
            move = input(str(prompt))
        return {"answer": move}
