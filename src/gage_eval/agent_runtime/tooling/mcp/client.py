from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class McpServerProcess:
    """Tracks scheduler-owned MCP server process state for one trial."""

    server_id: str
    client: Any
    owner_trial_id: str | None = None
    reset_count: int = 0

    def bind_trial(self, trial_id: str) -> None:
        self.owner_trial_id = trial_id

    def reset_for_trial(self, trial_id: str) -> None:
        self.owner_trial_id = trial_id
        self.reset_count += 1
        reset = getattr(self.client, "reset", None)
        if callable(reset):
            reset()
