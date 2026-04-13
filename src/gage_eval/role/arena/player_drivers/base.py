from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
from gage_eval.role.arena.core.players import BoundPlayer, PlayerBindingSpec


class PlayerDriver(ABC):
    driver_id: str
    family: str
    defaults: dict[str, object]

    def __init__(
        self,
        *,
        driver_id: str,
        family: str,
        defaults: Mapping[str, object] | None = None,
    ) -> None:
        self.driver_id = str(driver_id)
        self.family = str(family)
        self.defaults = dict(defaults or {})

    def resolve_params(self, spec: PlayerBindingSpec) -> dict[str, object]:
        merged = dict(self.defaults)
        merged.update(spec.driver_params)
        return merged

    @abstractmethod
    def bind(
        self,
        spec: PlayerBindingSpec,
        *,
        invocation: GameArenaInvocationContext | None = None,
    ) -> BoundPlayer:
        raise NotImplementedError
