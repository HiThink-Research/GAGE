"""Resolve game-kit runtime bindings for concrete samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from gage_eval.role.arena.core.errors import InvalidPlayerBindingError
from gage_eval.role.arena.core.players import PlayerBindingSpec, PlayerKind
from gage_eval.role.arena.player_drivers.registry import PlayerDriverRegistry
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.schedulers.registry import SchedulerRegistry
from gage_eval.role.arena.support.registry import SupportWorkflowRegistry
from gage_eval.game_kits.contracts import EnvSpec, GameKit, ResolvedRuntimeBinding
from gage_eval.game_kits.observation import ObservationWorkflowRegistry
from gage_eval.game_kits.registry import GameKitRegistry


class RuntimeBindingResolver:
    """Resolve game kits, envs, schedulers, and observation workflows."""

    _LEGACY_PLAYER_FIELDS = frozenset({"type", "ref", "moves"})

    def __init__(
        self,
        *,
        game_kits: GameKitRegistry,
        schedulers: SchedulerRegistry | None = None,
        observation_workflows: ObservationWorkflowRegistry | None = None,
        support_workflows: SupportWorkflowRegistry | None = None,
        player_drivers: PlayerDriverRegistry | None = None,
    ) -> None:
        self.game_kits = game_kits
        self.schedulers = schedulers or SchedulerRegistry(
            registry_view=game_kits.registry_view
        )
        self.observation_workflows = observation_workflows or ObservationWorkflowRegistry(
            registry_view=game_kits.registry_view
        )
        self.support_workflows = support_workflows or SupportWorkflowRegistry(
            registry_view=game_kits.registry_view
        )
        self.player_drivers = player_drivers or PlayerDriverRegistry(
            registry_view=game_kits.registry_view
        )

    def resolve(self, sample: ArenaSample) -> ResolvedRuntimeBinding:
        game_kit = self.game_kits.build(sample.game_kit)
        env_spec = self._resolve_env_spec(game_kit, sample)
        scheduler_binding = self._resolve_scheduler_binding(sample, game_kit, env_spec)
        observation_workflow_id = self._resolve_observation_workflow_id(game_kit, env_spec)
        support_workflow_id = self._resolve_support_workflow_id(game_kit)
        player_bindings = self._resolve_player_bindings(sample=sample, game_kit=game_kit)

        return ResolvedRuntimeBinding(
            game_kit=game_kit,
            env_spec=env_spec,
            scheduler=self.schedulers.build(scheduler_binding),
            resource_spec=env_spec.resource_spec,
            players=tuple(sample.players),
            player_bindings=player_bindings,
            player_driver_registry=self.player_drivers,
            observation_workflow=self.observation_workflows.build(observation_workflow_id),
            support_workflow=self.support_workflows.build(support_workflow_id),
        )

    def _resolve_player_bindings(
        self,
        *,
        sample: ArenaSample,
        game_kit: GameKit,
    ) -> tuple[PlayerBindingSpec, ...]:
        seat_spec = game_kit.seat_spec or {}
        known_seats = tuple(str(seat) for seat in seat_spec.get("seats", ()))
        bindings: list[PlayerBindingSpec] = []
        for index, raw_spec in enumerate(sample.players):
            if not isinstance(raw_spec, Mapping):
                raise InvalidPlayerBindingError("player specs must be mappings")
            self._reject_legacy_fields(raw_spec)
            seat = self._resolve_player_seat(raw_spec, index=index, known_seats=known_seats)
            player_kind = self._resolve_player_kind(raw_spec, seat=seat)
            driver_id = self._resolve_driver_id(raw_spec, player_kind=player_kind)
            player_id = str(raw_spec.get("player_id") or seat)
            backend_id = self._coerce_optional_text(raw_spec.get("backend_id"))
            agent_role_id = self._coerce_optional_text(raw_spec.get("agent_role_id"))
            actions = self._resolve_actions(raw_spec.get("actions"))
            driver_params = self._resolve_driver_params(raw_spec.get("driver_params"))
            self._validate_player_binding(
                seat=seat,
                player_kind=player_kind,
                player_id=player_id,
                backend_id=backend_id,
                actions=actions,
                known_seats=known_seats,
            )
            self.player_drivers.ensure_registered(driver_id)
            bindings.append(
                PlayerBindingSpec(
                    seat=seat,
                    player_id=player_id,
                    player_kind=player_kind,
                    driver_id=driver_id,
                    backend_id=backend_id,
                    agent_role_id=agent_role_id,
                    actions=actions,
                    driver_params=driver_params,
                )
            )
        return tuple(bindings)

    def _resolve_env_spec(self, game_kit: GameKit, sample: ArenaSample) -> EnvSpec:
        runtime_env = self._sample_env_override(sample)
        env_id = runtime_env or game_kit.default_env
        if env_id is None and len(game_kit.env_catalog) == 1:
            env_id = game_kit.env_catalog[0].env_id
        if env_id is None:
            raise KeyError(
                f"Game kit '{game_kit.kit_id}' does not define a default env and "
                "no runtime env override was provided"
            )
        return self._lookup_env_spec(game_kit, env_id)

    @staticmethod
    def _sample_env_override(sample: ArenaSample) -> str | None:
        if sample.env:
            return str(sample.env)
        overrides = sample.runtime_overrides or {}
        for key in ("env", "environment_id"):
            value = overrides.get(key)
            if value:
                return str(value)
        return None

    @staticmethod
    def _lookup_env_spec(game_kit: GameKit, env_id: str) -> EnvSpec:
        for env_spec in game_kit.env_catalog:
            if env_spec.env_id == env_id:
                return env_spec
        available_envs = ", ".join(env.env_id for env in game_kit.env_catalog) or "none"
        raise KeyError(
            f"Unknown env '{env_id}' for game kit '{game_kit.kit_id}'. "
            f"Available envs: {available_envs}"
        )

    @staticmethod
    def _resolve_scheduler_binding(
        sample: ArenaSample,
        game_kit: GameKit,
        env_spec: EnvSpec,
    ) -> str:
        sample_binding = RuntimeBindingResolver._sample_scheduler_override(sample)
        if sample_binding:
            return sample_binding
        return str(env_spec.scheduler_binding or game_kit.scheduler_binding)

    @staticmethod
    def _sample_scheduler_override(sample: ArenaSample) -> str | None:
        if sample.scheduler:
            return str(sample.scheduler)
        overrides = sample.runtime_overrides or {}
        for key in ("scheduler", "scheduler_binding"):
            value = overrides.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, Mapping):
                for nested_key in ("binding_id", "scheduler_binding", "id"):
                    nested = value.get(nested_key)
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()
        return None

    @staticmethod
    def _resolve_observation_workflow_id(game_kit: GameKit, env_spec: EnvSpec) -> str:
        return str(env_spec.observation_workflow or game_kit.observation_workflow)

    @staticmethod
    def _resolve_support_workflow_id(game_kit: GameKit) -> str:
        return str(game_kit.support_workflow or "arena/default")

    def _resolve_player_seat(
        self,
        raw_spec: Mapping[str, object],
        *,
        index: int,
        known_seats: tuple[str, ...],
    ) -> str:
        raw_seat = raw_spec.get("seat")
        if raw_seat is None:
            if index < len(known_seats):
                return known_seats[index]
            raise InvalidPlayerBindingError("player spec requires seat")
        return str(raw_seat)

    @staticmethod
    def _resolve_player_kind(
        raw_spec: Mapping[str, object],
        *,
        seat: str,
    ) -> PlayerKind:
        raw_kind = str(raw_spec.get("player_kind") or "").strip().lower()
        if raw_kind not in {"llm", "human", "agent", "dummy"}:
            raise InvalidPlayerBindingError(
                f"player '{seat}' requires supported player_kind"
            )
        return raw_kind  # type: ignore[return-value]

    def _resolve_driver_id(
        self,
        raw_spec: Mapping[str, object],
        *,
        player_kind: PlayerKind,
    ) -> str:
        explicit = raw_spec.get("driver")
        if explicit is not None and str(explicit).strip():
            return str(explicit).strip()
        return self.player_drivers.default_driver_id(player_kind)

    @staticmethod
    def _resolve_actions(raw_actions: object) -> tuple[str, ...] | None:
        if raw_actions is None:
            return None
        if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
            raise InvalidPlayerBindingError("dummy player actions must be a sequence")
        normalized = tuple(str(action) for action in raw_actions if action is not None)
        return normalized or None

    @staticmethod
    def _resolve_driver_params(raw_params: object) -> dict[str, object]:
        if raw_params is None:
            return {}
        if not isinstance(raw_params, Mapping):
            raise InvalidPlayerBindingError("driver_params must be a mapping")
        return {str(key): value for key, value in raw_params.items()}

    def _validate_player_binding(
        self,
        *,
        seat: str,
        player_kind: PlayerKind,
        player_id: str,
        backend_id: str | None,
        actions: tuple[str, ...] | None,
        known_seats: tuple[str, ...],
    ) -> None:
        if known_seats and seat not in known_seats:
            raise InvalidPlayerBindingError(
                f"player '{player_id}' uses unknown seat '{seat}'"
            )
        if player_kind == "llm" and backend_id is None:
            raise InvalidPlayerBindingError(
                f"llm player '{player_id}' requires backend_id"
            )
        if player_kind == "dummy" and not actions:
            raise InvalidPlayerBindingError(
                f"dummy player '{player_id}' requires non-empty actions"
            )

    def _reject_legacy_fields(self, raw_spec: Mapping[str, object]) -> None:
        legacy_fields = sorted(self._LEGACY_PLAYER_FIELDS.intersection(raw_spec))
        if legacy_fields:
            joined = ", ".join(legacy_fields)
            raise InvalidPlayerBindingError(
                f"legacy player fields are not supported: {joined}"
            )

    @staticmethod
    def _coerce_optional_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["PlayerDriverRegistry", "RuntimeBindingResolver"]
