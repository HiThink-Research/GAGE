"""Resolve game-kit runtime bindings for concrete samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from gage_eval.role.arena.core.errors import InvalidPlayerBindingError
from gage_eval.role.arena.core.players import PlayerBindingSpec, PlayerKind
from gage_eval.role.arena.player_drivers.registry import PlayerDriverRegistry
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.schedulers.registry import SchedulerRegistry
from gage_eval.role.arena.support.registry import SupportWorkflowRegistry
from gage_eval.game_kits.contracts import (
    EnvSpec,
    GameKit,
    HumanRealtimeInputProfile,
    RealtimeHumanControlProfile,
    ResolvedRuntimeBinding,
    ResolvedRuntimeProfile,
)
from gage_eval.game_kits.observation import ObservationWorkflowRegistry
from gage_eval.game_kits.registry import GameKitRegistry
from gage_eval.game_kits.visualization_specs import VisualizationSpecRegistry


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
        self.visualization_specs = VisualizationSpecRegistry(
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
        runtime_binding_policy = self._resolve_optional_runtime_ref(
            sample=sample,
            keys=("runtime_binding_policy",),
            env_value=env_spec.runtime_binding_policy,
            kit_value=game_kit.runtime_binding_policy,
        )
        scheduler_binding = self._resolve_scheduler_binding(sample, game_kit, env_spec)
        observation_workflow_id = self._resolve_observation_workflow_id(sample, game_kit, env_spec)
        game_display = self._resolve_game_display_ref(sample, game_kit, env_spec)
        replay_viewer = self._resolve_replay_viewer_ref(
            sample,
            game_kit,
            env_spec,
            game_display=game_display,
        )
        parser = self._resolve_optional_runtime_ref(
            sample=sample,
            keys=("parser",),
            env_value=env_spec.parser,
            kit_value=game_kit.parser,
        )
        renderer = self._resolve_optional_runtime_ref(
            sample=sample,
            keys=("renderer",),
            env_value=env_spec.renderer,
            kit_value=game_kit.renderer,
            system_default=self._resolve_renderer_ref(game_display),
        )
        replay_policy = self._resolve_optional_runtime_ref(
            sample=sample,
            keys=("replay_policy", "replay_adapter"),
            env_value=env_spec.replay_policy,
            kit_value=game_kit.replay_policy,
        )
        input_mapper = self._resolve_optional_runtime_ref(
            sample=sample,
            keys=("input_mapper",),
            env_value=env_spec.input_mapper,
            kit_value=game_kit.input_mapper,
        )
        game_content_refs = self._resolve_game_content_refs(sample, game_kit, env_spec)
        support_workflow_id = self._resolve_support_workflow_id(sample, game_kit, env_spec)
        player_bindings = self._resolve_player_bindings(sample=sample, game_kit=game_kit)
        scheduler = self.schedulers.build(scheduler_binding)
        runtime_profile = self._resolve_runtime_profile(
            sample=sample,
            scheduler_binding=scheduler_binding,
            scheduler=scheduler,
            player_bindings=player_bindings,
        )

        return ResolvedRuntimeBinding(
            game_kit=game_kit,
            env_spec=env_spec,
            scheduler=scheduler,
            resource_spec=env_spec.resource_spec,
            runtime_binding_policy=runtime_binding_policy,
            game_display=game_display,
            replay_viewer=replay_viewer,
            parser=parser,
            renderer=renderer,
            replay_policy=replay_policy,
            input_mapper=input_mapper,
            game_content_refs=game_content_refs,
            visualization_spec=self.visualization_specs.build(game_display),
            players=tuple(sample.players),
            player_bindings=player_bindings,
            player_driver_registry=self.player_drivers,
            observation_workflow=self.observation_workflows.build(observation_workflow_id),
            support_workflow=self.support_workflows.build(support_workflow_id),
            runtime_profile=runtime_profile,
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
    def _resolve_observation_workflow_id(
        sample: ArenaSample,
        game_kit: GameKit,
        env_spec: EnvSpec,
    ) -> str:
        return str(
            RuntimeBindingResolver._resolve_optional_runtime_ref(
                sample=sample,
                keys=("observation_workflow",),
                env_value=env_spec.observation_workflow,
                kit_value=game_kit.observation_workflow,
                system_default="arena/default",
            )
        )

    @staticmethod
    def _resolve_game_display_ref(
        sample: ArenaSample,
        game_kit: GameKit,
        env_spec: EnvSpec,
    ) -> str:
        return str(
            RuntimeBindingResolver._resolve_optional_runtime_ref(
                sample=sample,
                keys=("game_display", "game_display_descriptor", "visualization_spec"),
                env_value=env_spec.game_display,
                kit_value=game_kit.game_display or game_kit.visualization_spec,
                system_default="arena/default",
            )
        )

    @staticmethod
    def _resolve_replay_viewer_ref(
        sample: ArenaSample,
        game_kit: GameKit,
        env_spec: EnvSpec,
        *,
        game_display: str,
    ) -> str:
        return str(
            RuntimeBindingResolver._resolve_optional_runtime_ref(
                sample=sample,
                keys=("replay_viewer", "replay_viewer_descriptor"),
                env_value=env_spec.replay_viewer,
                kit_value=game_kit.replay_viewer,
                system_default=game_display,
            )
        )

    @staticmethod
    def _resolve_support_workflow_id(
        sample: ArenaSample,
        game_kit: GameKit,
        env_spec: EnvSpec,
    ) -> str:
        return str(
            RuntimeBindingResolver._resolve_optional_runtime_ref(
                sample=sample,
                keys=("support_workflow",),
                env_value=env_spec.defaults.get("support_workflow"),
                kit_value=game_kit.support_workflow,
                system_default="arena/default",
            )
        )

    def _resolve_renderer_ref(self, game_display: str | None) -> str | None:
        if game_display is None:
            return None
        try:
            spec = self.visualization_specs.build(game_display)
        except (KeyError, TypeError):
            return None
        return self._coerce_optional_text(spec.renderer_impl)

    @classmethod
    def _resolve_optional_runtime_ref(
        cls,
        *,
        sample: ArenaSample,
        keys: tuple[str, ...],
        env_value: object,
        kit_value: object,
        system_default: object = None,
    ) -> str | None:
        sample_value = cls._sample_runtime_override(sample, *keys)
        for candidate in (sample_value, env_value, kit_value, system_default):
            text = cls._coerce_optional_text(candidate)
            if text is not None:
                return text
        return None

    @staticmethod
    def _sample_runtime_override(sample: ArenaSample, *keys: str) -> object:
        overrides = sample.runtime_overrides or {}
        for key in keys:
            value = overrides.get(key)
            if value not in (None, ""):
                return value
        return None

    @classmethod
    def _resolve_game_content_refs(
        cls,
        sample: ArenaSample,
        game_kit: GameKit,
        env_spec: EnvSpec,
    ) -> dict[str, str]:
        merged: dict[str, str] = {}
        for source in (game_kit.game_content_refs, env_spec.game_content_refs):
            if not isinstance(source, Mapping):
                continue
            for key, value in source.items():
                text = cls._coerce_optional_text(value)
                if text is not None:
                    merged[str(key)] = text
        override_refs = cls._sample_runtime_override(sample, "game_content_refs", "content_refs")
        if isinstance(override_refs, Mapping):
            for key, value in override_refs.items():
                text = cls._coerce_optional_text(value)
                if text is not None:
                    merged[str(key)] = text
        return merged

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

    @classmethod
    def _resolve_runtime_profile(
        cls,
        *,
        sample: ArenaSample,
        scheduler_binding: str,
        scheduler: object,
        player_bindings: tuple[PlayerBindingSpec, ...],
    ) -> ResolvedRuntimeProfile:
        scheduler_family = cls._coerce_optional_text(getattr(scheduler, "family", None)) or "turn"
        human_realtime_inputs = cls._resolve_human_realtime_inputs(
            scheduler_family=scheduler_family,
            player_bindings=player_bindings,
        )
        session_tick_interval_ms = cls._resolve_session_tick_interval_ms(human_realtime_inputs)
        has_realtime_human_inputs = (
            scheduler_family == "real_time_tick"
            and bool(human_realtime_inputs)
            and session_tick_interval_ms is not None
        )
        pure_human_realtime = (
            scheduler_family == "real_time_tick"
            and bool(player_bindings)
            and all(binding.player_kind == "human" for binding in player_bindings)
        )
        realtime_human_control = cls._resolve_realtime_human_control(
            sample=sample,
            scheduler_family=scheduler_family,
            pure_human_realtime=pure_human_realtime,
            human_realtime_inputs=human_realtime_inputs,
            session_tick_interval_ms=session_tick_interval_ms,
        )
        if realtime_human_control is not None:
            session_tick_interval_ms = int(realtime_human_control.tick_interval_ms)
        scheduler_owns_realtime_clock = (
            realtime_human_control is not None
            and realtime_human_control.mode == "scheduler_owned_human_realtime"
            and realtime_human_control.activation_scope == "pure_human_only"
            and pure_human_realtime
            and session_tick_interval_ms is not None
        )
        supports_realtime_input = (
            has_realtime_human_inputs
            and cls._supports_scheduler_owned_realtime_input(realtime_human_control)
        )
        return ResolvedRuntimeProfile(
            scheduler_binding=str(scheduler_binding),
            scheduler_family=scheduler_family,
            tick_interval_ms=session_tick_interval_ms,
            pure_human_realtime=pure_human_realtime,
            scheduler_owns_realtime_clock=scheduler_owns_realtime_clock,
            supports_low_latency_realtime_input=supports_realtime_input,
            supports_realtime_input_websocket=supports_realtime_input,
            human_realtime_inputs=human_realtime_inputs,
            realtime_human_control=realtime_human_control,
        )

    @classmethod
    def _resolve_realtime_human_control(
        cls,
        *,
        sample: ArenaSample,
        scheduler_family: str,
        pure_human_realtime: bool,
        human_realtime_inputs: tuple[HumanRealtimeInputProfile, ...],
        session_tick_interval_ms: int | None,
    ) -> RealtimeHumanControlProfile | None:
        if scheduler_family != "real_time_tick" or not pure_human_realtime or not human_realtime_inputs:
            return None
        config = cls._resolve_realtime_human_policy_config(sample)
        if config is None:
            return None
        mode = cls._coerce_optional_text(config.get("mode"))
        activation_scope = cls._coerce_optional_text(config.get("activation_scope"))
        input_model = cls._coerce_optional_text(config.get("input_model"))
        if mode != "scheduler_owned_human_realtime":
            return None
        if activation_scope != "pure_human_only":
            return None
        if input_model not in {"continuous_state", "queued_command"}:
            return None
        tick_interval_ms = cls._coerce_positive_int(config.get("tick_interval_ms"))
        if tick_interval_ms is None:
            tick_interval_ms = session_tick_interval_ms
        if tick_interval_ms is None:
            return None
        frame_output_hz = cls._coerce_positive_int(config.get("frame_output_hz"))
        input_transport = cls._coerce_optional_text(config.get("input_transport"))
        artifact_sampling_mode = cls._coerce_optional_text(config.get("artifact_sampling_mode"))
        artifact_sampling_stride = cls._coerce_positive_int(config.get("artifact_sampling_stride"))
        snapshot_persist_stride = (
            cls._coerce_positive_int(config.get("snapshot_persist_stride"))
            or artifact_sampling_stride
        )
        if artifact_sampling_mode == "async_decimated_live" and snapshot_persist_stride is None:
            snapshot_persist_stride = 3
        if artifact_sampling_stride is None:
            artifact_sampling_stride = snapshot_persist_stride
        fallback_move = cls._coerce_optional_text(config.get("fallback_move"))
        max_commands_per_tick = cls._coerce_positive_int(config.get("max_commands_per_tick")) or 4
        max_command_queue_size = cls._coerce_positive_int(config.get("max_command_queue_size")) or 128
        command_stale_after_ms = cls._coerce_positive_int(config.get("command_stale_after_ms"))
        queue_overflow_policy = cls._coerce_optional_text(config.get("queue_overflow_policy"))
        bridge_stall_timeout_ms = cls._coerce_positive_int(config.get("bridge_stall_timeout_ms")) or 2000
        bridge_abort_timeout_ms = cls._coerce_positive_int(config.get("bridge_abort_timeout_ms"))
        if input_model == "queued_command":
            fallback_move = None
        elif fallback_move is None:
            fallback_move = cls._coerce_optional_text(human_realtime_inputs[0].timeout_fallback_move)
        return RealtimeHumanControlProfile(
            mode=mode,
            activation_scope=activation_scope,
            input_model=input_model,
            tick_interval_ms=tick_interval_ms,
            input_transport=input_transport,
            frame_output_hz=frame_output_hz,
            artifact_sampling_mode=artifact_sampling_mode,
            artifact_sampling_stride=artifact_sampling_stride,
            snapshot_persist_stride=snapshot_persist_stride,
            fallback_move=fallback_move,
            max_commands_per_tick=max_commands_per_tick,
            max_command_queue_size=max_command_queue_size,
            command_stale_after_ms=command_stale_after_ms,
            queue_overflow_policy=queue_overflow_policy,
            bridge_stall_timeout_ms=bridge_stall_timeout_ms,
            bridge_abort_timeout_ms=bridge_abort_timeout_ms,
        )

    @staticmethod
    def _supports_scheduler_owned_realtime_input(
        realtime_human_control: RealtimeHumanControlProfile | None,
    ) -> bool:
        if realtime_human_control is None:
            return False
        return (
            realtime_human_control.mode == "scheduler_owned_human_realtime"
            and realtime_human_control.activation_scope == "pure_human_only"
            and realtime_human_control.input_transport == "realtime_ws"
        )

    @staticmethod
    def _resolve_realtime_human_policy_config(sample: ArenaSample) -> Mapping[str, object] | None:
        overrides = dict(sample.runtime_overrides or {})
        candidates = (
            overrides.get("runtime_binding_policy_config"),
            overrides.get("realtime_human_policy"),
        )
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                return dict(candidate)
        runtime_binding_policy = overrides.get("runtime_binding_policy")
        if isinstance(runtime_binding_policy, Mapping):
            return dict(runtime_binding_policy)
        return None

    @classmethod
    def _resolve_human_realtime_inputs(
        cls,
        *,
        scheduler_family: str,
        player_bindings: tuple[PlayerBindingSpec, ...],
    ) -> tuple[HumanRealtimeInputProfile, ...]:
        if scheduler_family != "real_time_tick":
            return ()
        profiles: list[HumanRealtimeInputProfile] = []
        for binding in player_bindings:
            if binding.player_kind != "human":
                continue
            params = dict(binding.driver_params or {})
            tick_interval_ms = cls._resolve_player_tick_interval_ms(params)
            timeout_ms = cls._coerce_positive_int(params.get("timeout_ms"))
            timeout_fallback_move = cls._coerce_optional_text(params.get("timeout_fallback_move"))
            profiles.append(
                HumanRealtimeInputProfile(
                    player_id=str(binding.player_id),
                    semantics=cls._resolve_realtime_input_semantics(params),
                    tick_interval_ms=tick_interval_ms,
                    timeout_ms=timeout_ms,
                    timeout_fallback_move=timeout_fallback_move,
                )
            )
        return tuple(profiles)

    @classmethod
    def _resolve_realtime_input_semantics(cls, params: Mapping[str, object]) -> str:
        explicit = cls._coerce_optional_text(
            params.get("input_semantics") or params.get("realtime_input_semantics")
        )
        if explicit in {"continuous_state", "queued_command"}:
            return explicit
        if cls._coerce_optional_bool(params.get("stateful_actions"), default=False):
            return "continuous_state"
        return "queued_command"

    @classmethod
    def _resolve_session_tick_interval_ms(
        cls,
        profiles: tuple[HumanRealtimeInputProfile, ...],
    ) -> int | None:
        intervals = [
            int(profile.tick_interval_ms)
            for profile in profiles
            if profile.tick_interval_ms is not None and int(profile.tick_interval_ms) > 0
        ]
        if not intervals:
            return None
        return min(intervals)

    @classmethod
    def _resolve_player_tick_interval_ms(cls, params: Mapping[str, object]) -> int | None:
        tick_interval_ms = cls._coerce_positive_int(params.get("tick_interval_ms"))
        if tick_interval_ms is not None:
            return tick_interval_ms
        return cls._coerce_positive_int(params.get("timeout_ms"))

    @staticmethod
    def _coerce_optional_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_positive_int(value: object) -> int | None:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return None
        if normalized <= 0:
            return None
        return normalized

    @staticmethod
    def _coerce_optional_bool(value: object, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default


__all__ = ["PlayerDriverRegistry", "RuntimeBindingResolver"]
