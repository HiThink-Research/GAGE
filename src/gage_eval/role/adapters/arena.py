"""Arena role adapter for the unified GameArena runtime."""

from __future__ import annotations

import asyncio
from dataclasses import fields, is_dataclass
import functools
import inspect
from typing import Any, Dict, Mapping, Optional, Sequence

from gage_eval.assets.prompts.renderers import PromptRenderer
from gage_eval.evaluation.sample_ingress import resolve_runtime_sample_id
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState
from gage_eval.role.arena.core.bootstrap import build_gamearena_core
from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
from gage_eval.role.arena.core.types import ArenaSample


@registry.asset(
    "roles",
    "arena",
    desc="Arena role adapter for interactive games",
    tags=("role", "arena"),
    role_type="arena",
)
class ArenaRoleAdapter(RoleAdapter):
    """Role adapter that delegates all arena execution to GameArenaCore."""

    def __init__(
        self,
        adapter_id: str,
        *,
        environment: Optional[Dict[str, Any]] = None,
        rules: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any] | str] = None,
        parser: Optional[Dict[str, Any]] = None,
        visualizer: Optional[Dict[str, Any]] = None,
        human_input: Optional[Dict[str, Any]] = None,
        players: Optional[Sequence[Dict[str, Any]]] = None,
        game_kit: Optional[str] = None,
        env: Optional[str] = None,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        prompt_renderer: Optional[PromptRenderer] = None,
        registry_view=None,
        capabilities=(),
        role_type: str = "arena",
        **_,
    ) -> None:
        del environment, rules, parser, visualizer
        resolved_caps = tuple(capabilities) if capabilities else ("text",)
        super().__init__(
            adapter_id=adapter_id,
            role_type=role_type,
            capabilities=resolved_caps,
        )
        self._player_specs = list(players or [])
        self._human_input_cfg = dict(human_input or {})
        self._gamearena_defaults = {
            "game_kit": str(game_kit) if game_kit else None,
            "env": str(env) if env else None,
            "scheduler": self._normalize_gamearena_scheduler_binding(scheduler),
            "runtime_overrides": dict(runtime_overrides or {}),
        }
        self._prompt_renderer = prompt_renderer
        self._registry_view = registry_view
        self._gamearena_core = None

    def invoke(
        self,
        payload: Dict[str, Any],
        state: RoleAdapterState,
    ) -> Dict[str, Any]:
        return self._invoke_sync(payload, state)

    async def ainvoke(
        self,
        payload: Dict[str, Any],
        state: RoleAdapterState,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        func = functools.partial(self._invoke_sync, payload, state)
        return await loop.run_in_executor(None, func)

    def _invoke_sync(
        self,
        payload: Dict[str, Any],
        state: RoleAdapterState,
    ) -> Dict[str, Any]:
        del state
        sample = payload.get("sample") or {}
        trace = (
            payload.get("trace")
            if isinstance(payload.get("trace"), ObservabilityTrace)
            else None
        )
        role_manager = payload.get("role_manager") if isinstance(payload, dict) else None
        if not self._is_gamearena_sample(sample):
            raise ValueError(
                "Legacy arena runtime has been removed; configure game_kit on the sample or adapter"
            )
        return self._invoke_gamearena(
            sample,
            role_manager=role_manager,
            trace=trace,
        )

    def _invoke_gamearena(
        self,
        sample: Any,
        *,
        role_manager=None,
        trace: Optional[ObservabilityTrace] = None,
    ) -> Dict[str, Any]:
        core = self._get_gamearena_core()
        normalized_sample = self._normalize_gamearena_sample(sample)
        invocation_context = GameArenaInvocationContext(
            adapter_id=self.adapter_id,
            role_manager=role_manager,
            trace=trace,
            prompt_renderer=self._prompt_renderer,
            sample_payload=dict(sample) if isinstance(sample, Mapping) else {},
            human_input_config=dict(self._human_input_cfg),
        )
        run_sample = core.run_sample
        if _accepts_keyword(run_sample, "invocation_context"):
            result = run_sample(
                normalized_sample,
                invocation_context=invocation_context,
            )
        else:
            result = run_sample(normalized_sample)
        if isinstance(result, Mapping):
            return dict(result)
        if is_dataclass(result):
            return self._serialize_gamearena_value(result)
        return result

    def _get_gamearena_core(self):
        if self._gamearena_core is None:
            self._gamearena_core = build_gamearena_core(
                registry_view=self._registry_view,
            )
        return self._gamearena_core

    def _is_gamearena_sample(self, sample: Any) -> bool:
        if isinstance(sample, Mapping):
            return bool(sample.get("game_kit") or self._gamearena_defaults.get("game_kit"))
        return bool(getattr(sample, "game_kit", None) or self._gamearena_defaults.get("game_kit"))

    def _normalize_gamearena_sample(self, sample: Any) -> ArenaSample:
        if isinstance(sample, ArenaSample):
            if not self._gamearena_defaults.get("game_kit"):
                return sample
            sample_scheduler = sample.scheduler
            return ArenaSample(
                game_kit=sample.game_kit or str(self._gamearena_defaults["game_kit"]),
                env=sample.env or self._gamearena_defaults.get("env"),
                scheduler=sample_scheduler,
                players=sample.players or tuple(
                    dict(player) for player in self._player_specs if isinstance(player, Mapping)
                ),
                runtime_overrides=self._merge_gamearena_runtime_overrides(
                    sample.runtime_overrides,
                    sample_scheduler=sample_scheduler,
                ),
            )
        if not isinstance(sample, Mapping):
            raise TypeError("ArenaRoleAdapter requires a mapping or ArenaSample for GameArena")
        players = sample.get("players")
        if players is None:
            players = self._player_specs
        normalized_players = tuple(
            dict(player) for player in players if isinstance(player, Mapping)
        )
        sample_scheduler = self._normalize_gamearena_scheduler_binding(sample.get("scheduler"))
        runtime_overrides = self._merge_gamearena_runtime_overrides(
            sample.get("runtime_overrides"),
            sample_scheduler=sample_scheduler,
        )
        game_kit = sample.get("game_kit") or self._gamearena_defaults.get("game_kit")
        if not game_kit:
            raise KeyError("GameArena sample requires 'game_kit' in sample or adapter params")
        return ArenaSample(
            game_kit=str(game_kit),
            env=sample.get("env") or self._gamearena_defaults.get("env"),
            scheduler=sample_scheduler,
            players=normalized_players,
            runtime_overrides=runtime_overrides,
        )

    @staticmethod
    def _normalize_gamearena_scheduler_binding(
        scheduler: Mapping[str, Any] | str | None,
    ) -> str | None:
        if isinstance(scheduler, str) and scheduler.strip():
            return scheduler.strip()
        if isinstance(scheduler, Mapping):
            for key in ("binding_id", "scheduler_binding", "id"):
                value = scheduler.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _merge_gamearena_runtime_overrides(
        self,
        runtime_overrides: Mapping[str, Any] | None,
        *,
        sample_scheduler: str | None,
    ) -> dict[str, Any]:
        merged = dict(self._gamearena_defaults["runtime_overrides"])
        incoming = dict(runtime_overrides or {})
        if sample_scheduler is not None or any(
            key in incoming for key in ("scheduler", "scheduler_binding")
        ):
            merged.pop("scheduler", None)
            merged.pop("scheduler_binding", None)
        merged.update(incoming)
        default_scheduler = self._gamearena_defaults.get("scheduler")
        if (
            sample_scheduler is None
            and default_scheduler
            and "scheduler" not in incoming
            and "scheduler_binding" not in incoming
            and "scheduler" not in merged
            and "scheduler_binding" not in merged
        ):
            merged["scheduler"] = default_scheduler
        return merged

    @classmethod
    def _serialize_gamearena_value(cls, value: Any) -> Any:
        if is_dataclass(value):
            return {
                field.name: cls._serialize_gamearena_value(getattr(value, field.name))
                for field in fields(value)
            }
        if isinstance(value, Mapping):
            return {
                key: cls._serialize_gamearena_value(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(cls._serialize_gamearena_value(item) for item in value)
        if isinstance(value, list):
            return [cls._serialize_gamearena_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return tuple(cls._serialize_gamearena_value(item) for item in value)
        return value

    @staticmethod
    def _resolve_sample_id(sample: Dict[str, Any]) -> str:
        return resolve_runtime_sample_id(sample)


def _accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    if keyword in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


__all__ = ["ArenaRoleAdapter"]
