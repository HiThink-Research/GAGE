"""Arena domain providers for environment/runtime assembly."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from gage_eval.registry import registry
from gage_eval.role.arena.registry_loader import import_all_arena_asset_modules


class ArenaGameProvider:
    """Family-specific assembly hooks for arena games."""

    def matches(self, env_impl: str) -> bool:
        return False

    def enrich_environment_kwargs(
        self,
        *,
        env_impl: str,
        env_cfg: Mapping[str, Any],
        sample: Mapping[str, Any],
        trace: Any,
        chat_queue: Any,
        player_models: Optional[Mapping[str, str]],
        env_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return dict(env_kwargs)

    def resolve_renderer_impl(
        self,
        *,
        env_impl: str,
        visualizer_cfg: Mapping[str, Any],
        environment_cfg: Mapping[str, Any],
    ) -> Optional[str]:
        _ = env_impl, visualizer_cfg, environment_cfg
        return None

    def build_input_mapper(
        self,
        *,
        env_impl: str,
        environment_cfg: Mapping[str, Any],
        human_input_cfg: Mapping[str, Any],
    ) -> Any:
        _ = env_impl, environment_cfg, human_input_cfg
        return None


def _resolve_run_context(*, sample: Mapping[str, Any], trace: Any) -> tuple[Optional[str], Optional[str]]:
    run_id = trace.run_id if trace is not None else os.environ.get("GAGE_EVAL_RUN_ID")
    sample_id = (
        sample.get("id")
        or sample.get("sample_id")
        or os.environ.get("GAGE_EVAL_SAMPLE_ID")
    )
    return (str(run_id) if run_id else None, str(sample_id) if sample_id else None)


def _copy_selected_keys(
    *,
    source: Mapping[str, Any],
    target: dict[str, Any],
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        value = source.get(key)
        if value is not None:
            target[key] = value


def _coerce_bool_or_auto(value: Any, *, default: bool) -> bool:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return default
    return _coerce_bool(value, default=default)


def _normalize_recording_mode(value: Any) -> Optional[str]:
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode in {"action", "frame", "both"}:
        return mode
    return None


def _replay_mode_includes_frame(mode: str) -> bool:
    return mode in {"frame", "both"}


def _resolve_replay_recording_mode(replay_cfg: Any) -> str:
    if not isinstance(replay_cfg, Mapping):
        return "action"
    explicit_mode = _normalize_recording_mode(replay_cfg.get("mode"))
    if explicit_mode is not None:
        return explicit_mode

    action_cfg = replay_cfg.get("action")
    if not isinstance(action_cfg, Mapping):
        action_enabled = True
    else:
        action_enabled = _coerce_bool_or_auto(action_cfg.get("enabled"), default=True)

    frame_cfg = replay_cfg.get("frame")
    if isinstance(frame_cfg, Mapping):
        if "enabled" in frame_cfg:
            frame_enabled = _coerce_bool_or_auto(frame_cfg.get("enabled"), default=True)
        else:
            frame_enabled = True
    else:
        legacy_frame_cfg = replay_cfg.get("frame_capture")
        if isinstance(legacy_frame_cfg, Mapping):
            if "enabled" in legacy_frame_cfg:
                frame_enabled = _coerce_bool_or_auto(
                    legacy_frame_cfg.get("enabled"),
                    default=True,
                )
            else:
                frame_enabled = True
        else:
            frame_enabled = False

    if action_enabled and frame_enabled:
        return "both"
    if frame_enabled:
        return "frame"
    return "action"


@registry.asset(
    "arena_game_providers",
    "grid_board",
    desc="Grid-board arena provider for gomoku and tictactoe",
    tags=("arena", "provider", "grid"),
)
class GridBoardArenaGameProvider(ArenaGameProvider):
    def matches(self, env_impl: str) -> bool:
        normalized = str(env_impl).lower()
        return "gomoku" in normalized or "tictactoe" in normalized

    def enrich_environment_kwargs(
        self,
        *,
        env_impl: str,
        env_cfg: Mapping[str, Any],
        sample: Mapping[str, Any],
        trace: Any,
        chat_queue: Any,
        player_models: Optional[Mapping[str, str]],
        env_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        _ = sample, trace, chat_queue, player_models
        payload = dict(env_kwargs)
        _copy_selected_keys(source=env_cfg, target=payload, keys=("obs_image",))
        return payload

    def resolve_renderer_impl(
        self,
        *,
        env_impl: str,
        visualizer_cfg: Mapping[str, Any],
        environment_cfg: Mapping[str, Any],
    ) -> Optional[str]:
        _ = visualizer_cfg, environment_cfg
        if "tictactoe" in str(env_impl).lower():
            return "tictactoe_board_v1"
        return "gomoku_board_v1"

    def build_input_mapper(
        self,
        *,
        env_impl: str,
        environment_cfg: Mapping[str, Any],
        human_input_cfg: Mapping[str, Any],
    ) -> Any:
        _ = env_impl, human_input_cfg
        from gage_eval.role.arena.games.common.grid_coord_input_mapper import (
            GridCoordInputMapper,
        )

        action_schema = environment_cfg.get("action_schema")
        key_map = None
        enforce_legal_moves = True
        coord_scheme = environment_cfg.get("coord_scheme")
        if isinstance(action_schema, Mapping):
            key_map = action_schema.get("key_map")
            if action_schema.get("coord_scheme") is not None:
                coord_scheme = action_schema.get("coord_scheme")
            enforce_legal_moves = _coerce_bool(
                action_schema.get("enforce_legal_moves"),
                default=True,
            )
        return GridCoordInputMapper(
            key_map=key_map if isinstance(key_map, Mapping) else None,
            coord_scheme=str(coord_scheme) if coord_scheme else None,
            enforce_legal_moves=enforce_legal_moves,
        )


@registry.asset(
    "arena_game_providers",
    "pettingzoo",
    desc="PettingZoo arena provider",
    tags=("arena", "provider", "pettingzoo"),
)
class PettingZooArenaGameProvider(ArenaGameProvider):
    def matches(self, env_impl: str) -> bool:
        return "pettingzoo" in str(env_impl).lower()

    def enrich_environment_kwargs(
        self,
        *,
        env_impl: str,
        env_cfg: Mapping[str, Any],
        sample: Mapping[str, Any],
        trace: Any,
        chat_queue: Any,
        player_models: Optional[Mapping[str, str]],
        env_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        _ = env_impl, sample, trace, chat_queue, player_models
        payload = dict(env_kwargs)
        _copy_selected_keys(
            source=env_cfg,
            target=payload,
            keys=(
                "env_id",
                "env_kwargs",
                "seed",
                "action_labels",
                "use_action_meanings",
                "include_raw_obs",
                "agent_map",
            ),
        )
        return payload

    def build_input_mapper(
        self,
        *,
        env_impl: str,
        environment_cfg: Mapping[str, Any],
        human_input_cfg: Mapping[str, Any],
    ) -> Any:
        _ = env_impl, human_input_cfg
        from gage_eval.role.arena.games.pettingzoo.pettingzoo_input_mapper import (
            PettingZooDiscreteInputMapper,
        )

        action_schema = environment_cfg.get("action_schema")
        key_map = None
        enforce_legal_moves = True
        if isinstance(action_schema, Mapping):
            key_map = action_schema.get("key_map")
            enforce_legal_moves = _coerce_bool(
                action_schema.get("enforce_legal_moves"),
                default=True,
            )
        return PettingZooDiscreteInputMapper(
            key_map=key_map if isinstance(key_map, Mapping) else None,
            enforce_legal_moves=enforce_legal_moves,
        )


@registry.asset(
    "arena_game_providers",
    "vizdoom",
    desc="ViZDoom arena provider",
    tags=("arena", "provider", "vizdoom"),
)
class ViZDoomArenaGameProvider(ArenaGameProvider):
    _ENV_KEYS = (
        "use_single_process",
        "render_mode",
        "pov_view",
        "show_automap",
        "automap_scale",
        "automap_follow",
        "automap_stride",
        "show_pov",
        "capture_pov",
        "pov_stride",
        "allow_respawn",
        "respawn_grace_steps",
        "no_attack_seconds",
        "max_steps",
        "action_repeat",
        "sleep_s",
        "port",
        "config_path",
        "replay_output_dir",
        "game_id",
        "tick_rate_hz",
        "frame_stride",
        "time_source",
        "obs_image",
        "obs_image_history_len",
        "replay_in_env",
        "action_labels",
        "allow_partial_actions",
        "reset_retry_count",
        "death_check_warmup_steps",
    )

    def matches(self, env_impl: str) -> bool:
        return "vizdoom" in str(env_impl).lower()

    def enrich_environment_kwargs(
        self,
        *,
        env_impl: str,
        env_cfg: Mapping[str, Any],
        sample: Mapping[str, Any],
        trace: Any,
        chat_queue: Any,
        player_models: Optional[Mapping[str, str]],
        env_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        _ = env_impl, sample, chat_queue, player_models
        payload = dict(env_kwargs)
        _copy_selected_keys(source=env_cfg, target=payload, keys=self._ENV_KEYS)
        recording_mode = _resolve_replay_recording_mode(env_cfg.get("replay"))
        if _replay_mode_includes_frame(recording_mode):
            payload.setdefault("capture_pov", True)
        return payload

    def build_input_mapper(
        self,
        *,
        env_impl: str,
        environment_cfg: Mapping[str, Any],
        human_input_cfg: Mapping[str, Any],
    ) -> Any:
        _ = env_impl, human_input_cfg
        from gage_eval.role.arena.games.vizdoom.vizdoom_input_mapper import (
            ViZDoomInputMapper,
        )

        action_schema = environment_cfg.get("action_schema")
        key_map = None
        enforce_legal_moves = True
        if isinstance(action_schema, Mapping):
            key_map = action_schema.get("key_map")
            enforce_legal_moves = _coerce_bool(
                action_schema.get("enforce_legal_moves"),
                default=True,
            )
        return ViZDoomInputMapper(
            key_map=key_map if isinstance(key_map, Mapping) else None,
            enforce_legal_moves=enforce_legal_moves,
        )


@registry.asset(
    "arena_game_providers",
    "retro",
    desc="Retro arena provider",
    tags=("arena", "provider", "retro"),
)
class RetroArenaGameProvider(ArenaGameProvider):
    _ENV_KEYS = (
        "game",
        "state",
        "default_state",
        "rom_path",
        "runtime_policy",
        "display_mode",
        "record_bk2",
        "record_dir",
        "record_filename",
        "record_path",
        "action_mapping",
        "legal_moves",
        "info_feeder",
        "action_schema",
        "token_budget",
        "frame_stride",
        "snapshot_stride",
        "obs_image",
        "replay_output_dir",
        "replay_filename",
        "frame_output_dir",
        "seed",
    )

    def matches(self, env_impl: str) -> bool:
        return "retro" in str(env_impl).lower()

    def enrich_environment_kwargs(
        self,
        *,
        env_impl: str,
        env_cfg: Mapping[str, Any],
        sample: Mapping[str, Any],
        trace: Any,
        chat_queue: Any,
        player_models: Optional[Mapping[str, str]],
        env_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        _ = env_impl, chat_queue, player_models
        payload = dict(env_kwargs)
        _copy_selected_keys(source=env_cfg, target=payload, keys=self._ENV_KEYS)
        run_id, sample_id = _resolve_run_context(sample=sample, trace=trace)
        if run_id:
            payload["run_id"] = run_id
        if sample_id:
            payload["sample_id"] = sample_id
        return payload

    def build_input_mapper(
        self,
        *,
        env_impl: str,
        environment_cfg: Mapping[str, Any],
        human_input_cfg: Mapping[str, Any],
    ) -> Any:
        _ = env_impl
        from gage_eval.role.arena.games.retro.retro_input_mapper import (
            RetroInputMapper,
        )

        action_schema = environment_cfg.get("action_schema")
        hold_ticks_default = None
        if isinstance(action_schema, Mapping):
            hold_ticks_default = action_schema.get("hold_ticks_default")
        if hold_ticks_default is None:
            hold_ticks_default = human_input_cfg.get("hold_ticks_default")
        if hold_ticks_default is None:
            return RetroInputMapper()
        try:
            return RetroInputMapper(default_hold_ticks=int(hold_ticks_default))
        except (TypeError, ValueError):
            return RetroInputMapper()


@registry.asset(
    "arena_game_providers",
    "mahjong",
    desc="Mahjong arena provider",
    tags=("arena", "provider", "mahjong"),
)
class MahjongArenaGameProvider(ArenaGameProvider):
    def matches(self, env_impl: str) -> bool:
        return "mahjong" in str(env_impl).lower()

    def enrich_environment_kwargs(
        self,
        *,
        env_impl: str,
        env_cfg: Mapping[str, Any],
        sample: Mapping[str, Any],
        trace: Any,
        chat_queue: Any,
        player_models: Optional[Mapping[str, str]],
        env_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        _ = env_impl
        payload = dict(env_kwargs)
        run_id, sample_id = _resolve_run_context(sample=sample, trace=trace)
        if run_id:
            payload["run_id"] = run_id
        if sample_id:
            payload["sample_id"] = sample_id
        _copy_selected_keys(
            source=env_cfg,
            target=payload,
            keys=("chat_every_n", "replay_live", "replay_output_dir", "replay_filename"),
        )
        if chat_queue is not None:
            payload["chat_queue"] = chat_queue
        if player_models:
            payload["player_models"] = dict(player_models)
        return payload

    def resolve_renderer_impl(
        self,
        *,
        env_impl: str,
        visualizer_cfg: Mapping[str, Any],
        environment_cfg: Mapping[str, Any],
    ) -> Optional[str]:
        _ = env_impl, visualizer_cfg, environment_cfg
        return "mahjong_replay_v1"

    def build_input_mapper(
        self,
        *,
        env_impl: str,
        environment_cfg: Mapping[str, Any],
        human_input_cfg: Mapping[str, Any],
    ) -> Any:
        _ = env_impl, human_input_cfg
        from gage_eval.role.arena.games.mahjong.mahjong_input_mapper import (
            MahjongInputMapper,
        )

        action_schema = environment_cfg.get("action_schema")
        key_map = None
        enforce_legal_moves = True
        if isinstance(action_schema, Mapping):
            key_map = action_schema.get("key_map")
            enforce_legal_moves = _coerce_bool(
                action_schema.get("enforce_legal_moves"),
                default=True,
            )
        return MahjongInputMapper(
            key_map=key_map if isinstance(key_map, Mapping) else None,
            enforce_legal_moves=enforce_legal_moves,
        )


@registry.asset(
    "arena_game_providers",
    "doudizhu",
    desc="Doudizhu arena provider",
    tags=("arena", "provider", "doudizhu"),
)
class DoudizhuArenaGameProvider(ArenaGameProvider):
    def matches(self, env_impl: str) -> bool:
        return "doudizhu" in str(env_impl).lower()

    def enrich_environment_kwargs(
        self,
        *,
        env_impl: str,
        env_cfg: Mapping[str, Any],
        sample: Mapping[str, Any],
        trace: Any,
        chat_queue: Any,
        player_models: Optional[Mapping[str, str]],
        env_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        _ = env_impl, player_models
        payload = dict(env_kwargs)
        run_id, sample_id = _resolve_run_context(sample=sample, trace=trace)
        if run_id:
            payload["run_id"] = run_id
        if sample_id:
            payload["sample_id"] = sample_id
        _copy_selected_keys(
            source=env_cfg,
            target=payload,
            keys=(
                "chat_every_n",
                "replay_live",
                "replay_output_dir",
                "replay_filename",
                "context_include_public",
                "context_include_ui_state",
                "fast_finish_action",
                "fast_finish_human_only",
            ),
        )
        if chat_queue is not None:
            payload["chat_queue"] = chat_queue
        return payload

    def resolve_renderer_impl(
        self,
        *,
        env_impl: str,
        visualizer_cfg: Mapping[str, Any],
        environment_cfg: Mapping[str, Any],
    ) -> Optional[str]:
        _ = env_impl, visualizer_cfg, environment_cfg
        return "doudizhu_showdown_v1"

    def build_input_mapper(
        self,
        *,
        env_impl: str,
        environment_cfg: Mapping[str, Any],
        human_input_cfg: Mapping[str, Any],
    ) -> Any:
        _ = env_impl, human_input_cfg
        from gage_eval.role.arena.games.doudizhu.doudizhu_input_mapper import (
            DoudizhuInputMapper,
        )

        action_schema = environment_cfg.get("action_schema")
        key_map = None
        enforce_legal_moves = True
        if isinstance(action_schema, Mapping):
            key_map = action_schema.get("key_map")
            enforce_legal_moves = _coerce_bool(
                action_schema.get("enforce_legal_moves"),
                default=True,
            )
        return DoudizhuInputMapper(
            key_map=key_map if isinstance(key_map, Mapping) else None,
            enforce_legal_moves=enforce_legal_moves,
        )


class _DefaultArenaGameProvider(ArenaGameProvider):
    pass


_BUILTIN_PROVIDER_TYPES: tuple[type[ArenaGameProvider], ...] = (
    GridBoardArenaGameProvider,
    PettingZooArenaGameProvider,
    ViZDoomArenaGameProvider,
    RetroArenaGameProvider,
    MahjongArenaGameProvider,
    DoudizhuArenaGameProvider,
)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def resolve_arena_game_provider(env_impl: str) -> ArenaGameProvider:
    """Resolve one arena provider by env implementation family."""

    normalized_env_impl = str(env_impl or "").strip().lower()
    for provider_type in _BUILTIN_PROVIDER_TYPES:
        provider = provider_type()
        if provider.matches(normalized_env_impl):
            return provider
    import_all_arena_asset_modules("arena_game_providers")
    for entry in registry.list("arena_game_providers"):
        target = registry.get("arena_game_providers", entry.name)
        if isinstance(target, ArenaGameProvider):
            provider = target
        elif isinstance(target, type) and issubclass(target, ArenaGameProvider):
            provider = target()
        else:
            continue
        if provider.matches(normalized_env_impl):
            return provider
    return _DefaultArenaGameProvider()
