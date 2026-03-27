from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
from gage_eval.role.arena.core.players import BoundPlayer
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.support.context import SupportContext
from gage_eval.role.arena.support.hooks import SupportHook
from gage_eval.role.arena.types import ArenaAction


@dataclass
class GameSession:
    sample: ArenaSample
    environment: object | None = None
    player_specs: tuple[BoundPlayer, ...] = ()
    observation_workflow: object = None
    support_workflow: object = None
    max_steps: int = 256
    final_result: object | None = None
    tick: int = 0
    step: int = 0
    arena_trace: list[dict[str, object]] = field(default_factory=list)
    invocation_context: GameArenaInvocationContext | None = None
    _finalized: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_resolved(
        cls,
        sample,
        resolved,
        resources,
        *,
        invocation_context: GameArenaInvocationContext | None = None,
    ):
        player_specs = _bind_players(resolved=resolved, invocation_context=invocation_context)
        environment = _build_environment(
            sample=sample,
            resolved=resolved,
            resources=resources,
            player_specs=player_specs,
        )
        return cls(
            sample=sample,
            environment=environment,
            player_specs=player_specs,
            observation_workflow=resolved.observation_workflow,
            support_workflow=getattr(resolved, "support_workflow", None),
            max_steps=_resolve_max_steps(sample=sample, resolved=resolved),
            invocation_context=invocation_context,
        )

    def should_stop(self) -> bool:
        if self.final_result is not None:
            return True
        if self.environment is None:
            return True
        if self.step >= self.max_steps:
            self.final_result = self._build_result(result="max_steps", reason="max_steps")
            return True
        return bool(self.environment.is_terminal())

    def observe(self):
        if self.environment is None:
            raise RuntimeError("Game session environment is not initialized")
        player_id = str(self.environment.get_active_player())
        observation = self.environment.observe(player_id)
        workflow = self.observation_workflow
        if workflow is not None:
            observation = workflow.build(observation, self)
        support_context = self._run_support_hook(
            SupportHook.AFTER_OBSERVE,
            {"observation": observation, "player_id": player_id},
        )
        observation = support_context.payload.get("observation", observation)
        return observation

    def decide_current_player(self, observation) -> ArenaAction:
        if self.environment is None:
            raise RuntimeError("Game session environment is not initialized")
        player_id = str(self.environment.get_active_player())
        before_context = self._run_support_hook(
            SupportHook.BEFORE_DECIDE,
            {"observation": observation, "player_id": player_id},
        )
        observation = before_context.payload.get("observation", observation)
        player = self._require_player(player_id)
        action = player.next_action(observation)
        after_context = self._run_support_hook(
            SupportHook.AFTER_DECIDE,
            {
                "observation": observation,
                "player_id": player_id,
                "action": action.move,
                "action_object": action,
            },
        )
        updated_action = after_context.payload.get("action", action.move)
        return self._coerce_action(action, updated_action)

    def apply(self, action: ArenaAction) -> None:
        if self.environment is None:
            raise RuntimeError("Game session environment is not initialized")
        before_context = self._run_support_hook(
            SupportHook.BEFORE_APPLY,
            {
                "player_id": action.player,
                "action": action.move,
                "action_object": action,
            },
        )
        action = self._coerce_action(action, before_context.payload.get("action", action.move))
        result = self.environment.apply(action)
        if result is not None:
            self.final_result = result
        after_context = self._run_support_hook(
            SupportHook.AFTER_APPLY,
            {
                "player_id": action.player,
                "action": action.move,
                "action_object": action,
                "result": result,
            },
        )
        updated_result = after_context.payload.get("result", result)
        if updated_result is not None:
            self.final_result = updated_result
        self.arena_trace.append(
            {
                "tick": self.tick + 1,
                "step": self.step + 1,
                "player_id": action.player,
                "move": action.move,
                "result": getattr(result, "result", None),
                "winner": getattr(result, "winner", None),
            }
        )

    def advance(self) -> None:
        delta = self._resolve_progress_delta()
        self.tick += delta
        self.step += delta
        if self.final_result is None and self.environment is not None and self.environment.is_terminal():
            self.final_result = self._build_result(result="completed", reason="completed")

    def capture_output_tick(self) -> None:
        """Hook for schedulers that capture per-tick output on the v2 path."""
        return None

    def ensure_result(self) -> object | None:
        if self.final_result is not None:
            return self.final_result
        if self.environment is None:
            return None
        if self.environment.is_terminal():
            self.final_result = self._build_result(result="completed", reason="completed")
        return self.final_result

    def finalize(self) -> object | None:
        if self._finalized:
            return self.final_result
        self.ensure_result()
        final_context = self._run_support_hook(
            SupportHook.ON_FINALIZE,
            {"result": self.final_result, "sample": self.sample},
        )
        self.final_result = final_context.payload.get("result", self.final_result)
        self._finalized = True
        return self.final_result

    def _require_player(self, player_id: str) -> BoundPlayer:
        for player in self.player_specs:
            if player.player_id == player_id:
                return player
        available = ", ".join(player.player_id for player in self.player_specs) or "none"
        raise KeyError(f"Unknown active player '{player_id}'. Available players: {available}")

    def _build_result(self, *, result: str, reason: str | None) -> object | None:
        if self.environment is None:
            return None
        builder = getattr(self.environment, "build_result", None)
        if callable(builder):
            return builder(result=result, reason=reason)
        return {
            "winner": None,
            "result": result,
            "reason": reason,
            "move_count": self.step,
        }

    def _run_support_hook(self, hook: SupportHook, payload: dict[str, object]) -> SupportContext:
        context = SupportContext(
            payload=payload,
            state={
                "session": self,
                "hook": hook,
                "sample": self.sample,
                "environment": self.environment,
                "tick": self.tick,
                "step": self.step,
            },
        )
        workflow = self.support_workflow
        if workflow is None:
            return context
        runner = getattr(workflow, "run", None)
        if not callable(runner):
            return context
        result = runner(hook, context)
        if isinstance(result, SupportContext):
            return result
        return context

    @staticmethod
    def _coerce_action(action: ArenaAction, payload_action: object) -> ArenaAction:
        if payload_action is action.move:
            return action
        if isinstance(payload_action, ArenaAction):
            return payload_action
        if isinstance(payload_action, Mapping):
            player = str(payload_action.get("player", action.player))
            move = payload_action.get("move", action.move)
            raw = payload_action.get("raw", action.raw)
            metadata = payload_action.get("metadata", action.metadata)
            return ArenaAction(
                player=player,
                move=move,  # type: ignore[arg-type]
                raw=raw,  # type: ignore[arg-type]
                metadata=dict(metadata) if isinstance(metadata, Mapping) else action.metadata,
            )
        return ArenaAction(
            player=action.player,
            move=payload_action,  # type: ignore[arg-type]
            raw=action.raw,
            metadata=action.metadata,
        )

    def _resolve_progress_delta(self) -> int:
        if self.environment is None:
            return 1
        resolver = getattr(self.environment, "consume_session_progress_delta", None)
        if not callable(resolver):
            return 1
        try:
            delta = int(resolver())
        except Exception:
            return 1
        return max(0, delta)


def _resolve_max_steps(*, sample: ArenaSample, resolved) -> int:
    runtime_overrides = sample.runtime_overrides or {}
    for key in ("max_steps", "max_turns", "max_ticks"):
        value = runtime_overrides.get(key)
        if value is not None:
            return max(1, int(value))
    defaults = getattr(resolved.scheduler, "defaults", {}) or {}
    for key in ("max_steps", "max_turns", "max_ticks"):
        value = defaults.get(key)
        if value is not None:
            return max(1, int(value))
    return 256


def _build_environment(*, sample: ArenaSample, resolved, resources, player_specs) -> object:
    env_factory = resolved.env_spec.defaults.get("env_factory")
    if not callable(env_factory):
        raise KeyError(
            f"Env '{resolved.env_spec.env_id}' for game kit '{resolved.game_kit.kit_id}' "
            "does not define a callable 'env_factory'"
        )
    return env_factory(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
    )


def _bind_players(
    *,
    resolved,
    invocation_context: GameArenaInvocationContext | None,
) -> tuple[BoundPlayer, ...]:
    bindings = tuple(getattr(resolved, "player_bindings", ()) or ())
    if not bindings:
        return ()
    registry = getattr(resolved, "player_driver_registry", None)
    if registry is None:
        raise RuntimeError("resolved runtime binding is missing player_driver_registry")
    return tuple(registry.bind_all(bindings, invocation=invocation_context))
