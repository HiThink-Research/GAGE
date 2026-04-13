from __future__ import annotations

from dataclasses import dataclass

from gage_eval.role.arena.core.errors import PlayerExecutionUnavailableError
from gage_eval.role.arena.core.players import BaseBoundPlayer, PlayerBindingSpec
from gage_eval.role.arena.player_drivers.base import PlayerDriver


@dataclass
class AgentRoleStubBoundPlayer(BaseBoundPlayer):
    agent_role_id: str | None = None

    def next_action(self, observation):
        del observation
        agent_label = self.agent_role_id or self.player_id
        raise PlayerExecutionUnavailableError(
            f"Agent player '{agent_label}' is not implemented in the new runtime yet"
        )


class AgentRoleStubDriver(PlayerDriver):
    def bind(self, spec: PlayerBindingSpec, *, invocation=None) -> AgentRoleStubBoundPlayer:
        del invocation
        return AgentRoleStubBoundPlayer(
            player_id=spec.player_id,
            display_name=spec.player_id,
            seat=spec.seat,
            player_kind=spec.player_kind,
            agent_role_id=spec.agent_role_id,
            metadata={
                "driver_id": self.driver_id,
                "seat": spec.seat,
                "agent_role_id": spec.agent_role_id,
            },
        )
