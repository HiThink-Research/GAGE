from __future__ import annotations

from typing import Any

from gage_eval.role.arena.core.errors import PlayerDriverLookupError
from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
from gage_eval.role.arena.core.players import BoundPlayer, PlayerBindingSpec, PlayerKind
from gage_eval.role.arena.player_drivers.agent_role_stub import AgentRoleStubDriver
from gage_eval.role.arena.player_drivers.base import PlayerDriver
from gage_eval.role.arena.player_drivers.dummy import DummyPlayerDriver
from gage_eval.role.arena.player_drivers.human_local_input import LocalHumanInputDriver
from gage_eval.role.arena.player_drivers.llm_backend import LLMBackendDriver
from gage_eval.role.arena.support import specs as _support_specs  # noqa: F401
from gage_eval.registry import registry

_DEFAULT_DRIVER_IDS: dict[PlayerKind, str] = {
    "dummy": "player_driver/dummy",
    "human": "player_driver/human_local_input",
    "llm": "player_driver/llm_backend",
    "agent": "player_driver/agent_role_stub",
}

_DRIVER_IMPLS = {
    "dummy": DummyPlayerDriver,
    "human_local_input": LocalHumanInputDriver,
    "llm_backend": LLMBackendDriver,
    "agent_role_stub": AgentRoleStubDriver,
}


def build_driver_from_spec(spec) -> PlayerDriver:
    impl = str(getattr(spec, "impl", "") or "").strip()
    family = str(getattr(spec, "family", "") or "").strip()
    driver_cls = _DRIVER_IMPLS.get(impl) or _DRIVER_IMPLS.get(family)
    if driver_cls is None:
        raise PlayerDriverLookupError(
            f"Unknown player driver implementation '{impl or family}' for '{spec.driver_id}'"
        )
    return driver_cls(
        driver_id=str(spec.driver_id),
        family=family or impl,
        defaults=getattr(spec, "defaults", {}) or {},
    )


class PlayerDriverRegistry:
    def __init__(self, *, registry_view=None) -> None:
        self._registry = registry_view or registry
        self.registry_view = self._registry

    def default_driver_id(self, player_kind: PlayerKind) -> str:
        return _DEFAULT_DRIVER_IDS[player_kind]

    def build(self, driver_id: str) -> PlayerDriver:
        asset = self._registry.get("player_drivers", driver_id)
        if isinstance(asset, PlayerDriver):
            return asset
        if callable(asset):
            asset = asset()
        return build_driver_from_spec(asset)

    def ensure_registered(self, driver_id: str) -> None:
        self.build(driver_id)

    def bind(
        self,
        binding: PlayerBindingSpec,
        *,
        invocation: GameArenaInvocationContext | None = None,
    ) -> BoundPlayer:
        driver = self.build(binding.driver_id)
        return driver.bind(binding, invocation=invocation)

    def bind_all(
        self,
        bindings: tuple[PlayerBindingSpec, ...],
        *,
        invocation: GameArenaInvocationContext | None = None,
    ) -> tuple[BoundPlayer, ...]:
        return tuple(self.bind(binding, invocation=invocation) for binding in bindings)


__all__ = ["PlayerDriverRegistry", "build_driver_from_spec"]
