from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GameArenaInvocationContext:
    adapter_id: str | None = None
    role_manager: Any | None = None
    trace: Any | None = None
    prompt_renderer: Any | None = None
    sample_payload: Mapping[str, Any] = field(default_factory=dict)
    player_action_queues: Mapping[str, Any] = field(default_factory=dict)
    human_input_config: Mapping[str, Any] = field(default_factory=dict)

    def queue_for_player(self, player_id: str) -> Any | None:
        return self.player_action_queues.get(str(player_id))
