"""Stub-first OpenRA arena environment for CI-safe GameKit integration."""

from __future__ import annotations

import base64
import json
import os
import struct
import time
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

from gage_eval.game_kits.real_time_game.backend_mode import normalize_backend_mode
from gage_eval.game_kits.real_time_game.openra.native_bridge import (
    DEFAULT_FRAME_RATE_HZ,
    OpenRANativeBridge,
)
from gage_eval.role.arena.replay_paths import resolve_invocation_run_sample_ids
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult

DEFAULT_PLAYER_IDS = ("player_0", "player_1")
DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
DEFAULT_STREAM_ID = "main"
DEFAULT_OBJECTIVES = (
    {"id": "hold_ridge", "label": "Hold the ridge", "status": "active"},
    {"id": "destroy_radar", "label": "Destroy radar dome", "status": "pending"},
)
DEFAULT_UNIT_LAYOUT = {
    "player_0": (
        {
            "id": "mcv_1",
            "owner": "player_0",
            "label": "MCV",
            "kind": "vehicle",
            "hp": 95,
            "status": "idle",
            "position": {"x": 12, "y": 4},
            "selected": True,
        },
        {
            "id": "rifle_2",
            "owner": "player_0",
            "label": "Rifle Infantry",
            "kind": "infantry",
            "hp": 82,
            "status": "moving",
            "position": {"x": 14, "y": 7},
            "selected": True,
        },
    ),
    "player_1": (
        {
            "id": "tank_7",
            "owner": "player_1",
            "label": "Medium Tank",
            "kind": "vehicle",
            "hp": 88,
            "status": "guarding",
            "position": {"x": 24, "y": 18},
            "selected": False,
        },
        {
            "id": "rocket_4",
            "owner": "player_1",
            "label": "Rocket Soldier",
            "kind": "infantry",
            "hp": 74,
            "status": "holding",
            "position": {"x": 22, "y": 17},
            "selected": False,
        },
    ),
}
DEFAULT_SELECTIONS = {
    "player_0": ("mcv_1", "rifle_2"),
    "player_1": ("tank_7",),
}
DEFAULT_NATIVE_STARTUP_ORDERS = (
    "option fog False",
    "option explored True",
)
NATIVE_SCRIPT_BASE_VIEWPORT = {"width": 1280, "height": 720}
CAMERA_TOUR_NATIVE_EVENTS = (
    {"event_type": "key_down", "key": "ArrowRight", "modifiers": []},
    None,
    None,
    None,
    None,
    None,
    None,
    {"event_type": "key_up", "key": "ArrowRight", "modifiers": []},
    {"event_type": "key_down", "key": "ArrowDown", "modifiers": []},
    None,
    None,
    None,
    None,
    None,
    None,
    {"event_type": "key_up", "key": "ArrowDown", "modifiers": []},
    {"event_type": "key_down", "key": "ArrowLeft", "modifiers": []},
    None,
    None,
    None,
    None,
    None,
    None,
    {"event_type": "key_up", "key": "ArrowLeft", "modifiers": []},
    {"event_type": "key_down", "key": "ArrowUp", "modifiers": []},
    None,
    None,
    None,
    None,
    None,
    None,
    {"event_type": "key_up", "key": "ArrowUp", "modifiers": []},
)
UNIT_PATROL_NATIVE_EVENTS = (
    {"event_type": "mouse_down", "button": "left", "x": 724, "y": 380},
    {"event_type": "mouse_up", "button": "left", "x": 724, "y": 380},
    {"event_type": "mouse_down", "button": "right", "x": 920, "y": 332},
    {"event_type": "mouse_up", "button": "right", "x": 920, "y": 332},
    {"event_type": "mouse_down", "button": "right", "x": 944, "y": 520},
    {"event_type": "mouse_up", "button": "right", "x": 944, "y": 520},
    {"event_type": "mouse_down", "button": "right", "x": 612, "y": 564},
    {"event_type": "mouse_up", "button": "right", "x": 612, "y": 564},
    {"event_type": "mouse_down", "button": "right", "x": 468, "y": 320},
    {"event_type": "mouse_up", "button": "right", "x": 468, "y": 320},
)
DEFAULT_PRODUCTION = {
    "player_0": (
        {
            "building_id": "barracks_1",
            "label": "Barracks",
            "items": (
                {"id": "ranger", "label": "Ranger", "progress": 0.4},
            ),
        },
    ),
    "player_1": (
        {
            "building_id": "war_factory_1",
            "label": "War Factory",
            "items": (
                {"id": "artillery", "label": "Artillery", "progress": 0.25},
            ),
        },
    ),
}
COMMAND_TARGETS = (
    {"x": 18, "y": 11},
    {"x": 19, "y": 11},
    {"x": 20, "y": 12},
    {"x": 21, "y": 12},
)
PAN_DIRECTIONS = ("left", "right", "up", "down")
QUEUE_ROTATION = (
    {"id": "ranger", "label": "Ranger"},
    {"id": "rocket_soldier", "label": "Rocket Soldier"},
    {"id": "medic", "label": "Medic"},
)
OPENRA_REFERENCE_ROOT_ENV = "GAGE_OPENRA_REFERENCE_ROOT"
OPENRA_PREVIEW_OVERRIDES = {
    "ra_map01": Path("mods/ra/maps/marigold-town.oramap"),
}


@dataclass(frozen=True)
class OpenRAResolvedAction:
    action_id: str
    payload: dict[str, Any]
    raw_text: str


@dataclass(frozen=True)
class OpenRAReferencePreview:
    title: str
    map_size: dict[str, int] | None
    bounds: dict[str, int] | None
    image_size: dict[str, int] | None
    mime_type: str
    data_url: str
    preview_source: str = "reference_map_preview"


def _locate_openra_reference_root() -> Path | None:
    env_override = os.environ.get(OPENRA_REFERENCE_ROOT_ENV)
    candidate_paths: list[Path] = []
    if env_override:
        candidate_paths.append(Path(env_override).expanduser())
    for ancestor in Path(__file__).resolve().parents:
        candidate_paths.append(ancestor / "reference" / "gamearena" / "OpenRA-bleed")
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()
    return None


def _parse_openra_pair(raw_value: str | None) -> tuple[int, int] | None:
    if raw_value is None:
        return None
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _parse_openra_quad(raw_value: str | None) -> tuple[int, int, int, int] | None:
    if raw_value is None:
        return None
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 4:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None


def _parse_map_yaml_metadata(raw_yaml: str) -> tuple[str | None, dict[str, int] | None, dict[str, int] | None]:
    title: str | None = None
    map_size: dict[str, int] | None = None
    bounds: dict[str, int] | None = None
    for raw_line in raw_yaml.splitlines():
        line = raw_line.strip()
        if line.startswith("Title:"):
            title = line.split(":", 1)[1].strip() or None
            continue
        if line.startswith("MapSize:"):
            parsed_size = _parse_openra_pair(line.split(":", 1)[1].strip())
            if parsed_size is not None:
                map_size = {"width": parsed_size[0], "height": parsed_size[1]}
            continue
        if line.startswith("Bounds:"):
            parsed_bounds = _parse_openra_quad(line.split(":", 1)[1].strip())
            if parsed_bounds is not None:
                bounds = {
                    "x": parsed_bounds[0],
                    "y": parsed_bounds[1],
                    "width": parsed_bounds[2],
                    "height": parsed_bounds[3],
                }
    return title, map_size, bounds


def _normalize_string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    resolved: list[str] = []
    for item in value:
        normalized = str(item or "").strip()
        if normalized:
            resolved.append(normalized)
    return tuple(resolved)


def _normalize_native_launch_mode(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"skirmish", "mission", "local"}:
        return normalized
    return None


def _infer_native_launch_mode(*, env_id: str, map_ref: str | None) -> str | None:
    normalized_env_id = str(env_id or "").strip().lower()
    normalized_map_ref = str(map_ref or "").strip().lower()
    if "skirmish" in normalized_env_id or ".oramap" in normalized_map_ref:
        return "skirmish"
    return None


def _scale_native_demo_coordinate(value: int, *, axis: str, viewport: Mapping[str, Any]) -> int:
    base = int(NATIVE_SCRIPT_BASE_VIEWPORT["width" if axis == "x" else "height"])
    target = int(viewport.get("width" if axis == "x" else "height", base))
    return int(round((int(value) / base) * target))


def _read_png_dimensions(image_bytes: bytes) -> dict[str, int] | None:
    if len(image_bytes) < 24 or image_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", image_bytes[16:24])
    return {"width": int(width), "height": int(height)}


def _resolve_preview_asset_path(
    *,
    reference_root: Path,
    env_id: str,
    mod_id: str | None,
    map_ref: str | None,
) -> Path | None:
    override = OPENRA_PREVIEW_OVERRIDES.get(env_id)
    if override is not None:
        candidate = reference_root / override
        if candidate.exists():
            return candidate
    if not map_ref:
        return None
    normalized_segments = [segment for segment in str(map_ref).split("/") if segment]
    if not normalized_segments:
        return None
    resolved_mod_id = (
        str(mod_id).strip()
        if mod_id is not None and str(mod_id).strip()
        else str(normalized_segments[-2]).strip() if len(normalized_segments) >= 2 else ""
    )
    map_name = normalized_segments[-1]
    if not resolved_mod_id:
        return None
    candidate = reference_root / "mods" / resolved_mod_id / "maps" / map_name
    if candidate.exists():
        return candidate
    return None


def _load_preview_bundle(asset_path: Path) -> tuple[bytes | None, str | None]:
    if asset_path.is_dir():
        image_path = asset_path / "map.png"
        yaml_path = asset_path / "map.yaml"
        if not image_path.exists() or not yaml_path.exists():
            return None, None
        return image_path.read_bytes(), yaml_path.read_text(encoding="utf-8")
    if asset_path.suffix.lower() == ".oramap":
        with zipfile.ZipFile(asset_path) as archive:
            try:
                return (
                    archive.read("map.png"),
                    archive.read("map.yaml").decode("utf-8"),
                )
            except KeyError:
                return None, None
    return None, None


def _resolve_reference_preview(
    *,
    env_id: str,
    mod_id: str | None,
    map_ref: str | None,
) -> OpenRAReferencePreview | None:
    reference_root = _locate_openra_reference_root()
    if reference_root is None:
        return None
    asset_path = _resolve_preview_asset_path(
        reference_root=reference_root,
        env_id=env_id,
        mod_id=mod_id,
        map_ref=map_ref,
    )
    if asset_path is None:
        return None
    image_bytes, raw_yaml = _load_preview_bundle(asset_path)
    if image_bytes is None or raw_yaml is None:
        return None
    title, map_size, bounds = _parse_map_yaml_metadata(raw_yaml)
    data_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    return OpenRAReferencePreview(
        title=title or env_id,
        map_size=map_size,
        bounds=bounds,
        image_size=_read_png_dimensions(image_bytes),
        mime_type="image/png",
        data_url=data_url,
    )


class OpenRAArenaEnvironment:
    """Deterministic RTS-flavored environment used by the OpenRA GameKit."""

    def __init__(
        self,
        *,
        env_id: str = "ra_map01",
        backend_mode: str = "dummy",
        player_ids: Sequence[str] | None = None,
        player_names: Mapping[str, str] | None = None,
        max_decisions: int = 6,
        run_id: str | None = None,
        sample_id: str | None = None,
        seed: int | None = None,
        stream_id: str = DEFAULT_STREAM_ID,
        viewport: Mapping[str, Any] | None = None,
        mod_id: str | None = None,
        map_ref: str | None = None,
        native_bridge: Any | None = None,
        native_step_interval_s: float = 0.25,
        native_frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
        native_launch_mode: str | None = None,
        native_startup_orders: Sequence[str] | None = None,
        native_demo_script: str | None = None,
        native_demo_script_mode: str = "noop",
        **_: Any,
    ) -> None:
        self._env_id = str(env_id or "ra_map01")
        self._backend_mode = normalize_backend_mode(backend_mode, default="dummy")
        self._mod_id = str(mod_id).strip() if mod_id is not None and str(mod_id).strip() else None
        self._map_ref = str(map_ref).strip() if map_ref is not None and str(map_ref).strip() else None
        resolved_player_ids = tuple(str(item) for item in (player_ids or DEFAULT_PLAYER_IDS))
        self._player_ids = resolved_player_ids or DEFAULT_PLAYER_IDS
        self._player_names = {
            str(player_id): str((player_names or {}).get(player_id, player_id))
            for player_id in self._player_ids
        }
        self._max_decisions = max(1, int(max_decisions))
        self._run_id = run_id
        self._sample_id = sample_id
        self._seed = seed
        self._stream_id = str(stream_id or DEFAULT_STREAM_ID)
        resolved_viewport = dict(viewport or DEFAULT_VIEWPORT)
        self._viewport = {
            "width": int(resolved_viewport.get("width", DEFAULT_VIEWPORT["width"])),
            "height": int(resolved_viewport.get("height", DEFAULT_VIEWPORT["height"])),
        }
        self._native_launch_mode = _normalize_native_launch_mode(native_launch_mode) or _infer_native_launch_mode(
            env_id=self._env_id,
            map_ref=self._map_ref,
        )
        self._native_startup_orders = _normalize_string_sequence(native_startup_orders)
        self._native_demo_script = str(native_demo_script or "").strip().lower() or None
        self._native_demo_script_mode = (
            "live_stream"
            if str(native_demo_script_mode or "").strip().lower() == "live_stream"
            else "noop"
        )
        self._native_demo_step = 0
        self._native_bridge = None
        if self._backend_mode == "native":
            self._native_bridge = (
                native_bridge
                if native_bridge is not None
                else OpenRANativeBridge(
                    env_id=self._env_id,
                    mod_id=self._mod_id or "cnc",
                    map_ref=self._map_ref,
                    viewport=self._viewport,
                    launch_mode=self._native_launch_mode,
                    frame_rate_hz=float(native_frame_rate_hz),
                    startup_orders=self._native_startup_orders,
                )
            )
        self._native_step_interval_s = max(0.05, float(native_step_interval_s))
        self._native_last_apply_monotonic = 0.0
        self._native_last_demo_pump_monotonic = 0.0
        self._native_state: dict[str, Any] = {}
        self._reference_preview = _resolve_reference_preview(
            env_id=self._env_id,
            mod_id=self._mod_id,
            map_ref=self._map_ref,
        )

        self._active_player_index = 0
        self._tick = 0
        self._step = 0
        self._decision_count = 0
        self._illegal_move_count = 0
        self._last_move: str | None = None
        self._last_reward: float | None = None
        self._terminal = False
        self._final_result: GameResult | None = None
        self._last_frame: dict[str, Any] = {}
        self._move_log: list[dict[str, Any]] = []
        self._objectives: list[dict[str, Any]] = []
        self._units_by_player: dict[str, list[dict[str, Any]]] = {}
        self._selection_by_player: dict[str, list[str]] = {}
        self._production_by_player: dict[str, list[dict[str, Any]]] = {}
        self._economy_by_player: dict[str, dict[str, Any]] = {}
        self.reset()

    @classmethod
    def from_runtime(
        cls,
        *,
        sample,
        resolved,
        resources,
        player_specs,
        invocation_context=None,
    ) -> "OpenRAArenaEnvironment":
        del resources
        defaults = {
            **dict(resolved.game_kit.defaults),
            **dict(resolved.env_spec.defaults),
            **dict(sample.runtime_overrides or {}),
        }
        backend_mode = str(defaults.get("backend_mode", "dummy"))
        run_id, sample_id = resolve_invocation_run_sample_ids(
            invocation_context=invocation_context,
            run_id=defaults.get("run_id"),
            sample_id=defaults.get("sample_id"),
        )
        player_ids = [str(getattr(player, "player_id")) for player in player_specs]
        player_names = {
            str(getattr(player, "player_id")): str(getattr(player, "display_name"))
            for player in player_specs
        }
        return cls(
            env_id=str(defaults.get("env_id") or resolved.env_spec.env_id or "ra_map01"),
            backend_mode=backend_mode,
            player_ids=player_ids or DEFAULT_PLAYER_IDS,
            player_names=player_names,
            max_decisions=int(
                defaults.get("max_decisions")
                or defaults.get("max_turns")
                or (
                    defaults.get("stub_max_ticks")
                    if backend_mode != "native"
                    else None
                )
                or defaults.get("stub_max_ticks")
                or 6
            ),
            run_id=run_id,
            sample_id=sample_id,
            seed=defaults.get("seed"),
            stream_id=str(defaults.get("stream_id", DEFAULT_STREAM_ID)),
            viewport=defaults.get("viewport"),
            mod_id=defaults.get("mod_id"),
            map_ref=defaults.get("map_ref"),
            native_step_interval_s=float(defaults.get("native_step_interval_s", 0.25)),
            native_frame_rate_hz=float(defaults.get("native_frame_rate_hz", DEFAULT_FRAME_RATE_HZ)),
            native_launch_mode=defaults.get("native_launch_mode"),
            native_startup_orders=defaults.get("native_startup_orders"),
            native_demo_script=defaults.get("native_demo_script"),
            native_demo_script_mode=str(defaults.get("native_demo_script_mode", "noop")),
        )

    def reset(self) -> None:
        self._active_player_index = 0
        self._tick = 0
        self._step = 0
        self._decision_count = 0
        self._illegal_move_count = 0
        self._last_move = None
        self._last_reward = None
        self._terminal = False
        self._final_result = None
        self._move_log = []
        self._objectives = []
        self._units_by_player = {}
        self._selection_by_player = {}
        self._production_by_player = {}
        self._economy_by_player = {}
        self._native_state = {}
        self._native_demo_step = 0
        if self._backend_mode == "native":
            for player_id in self._player_ids:
                self._units_by_player[player_id] = []
                self._selection_by_player[player_id] = []
                self._production_by_player[player_id] = []
                self._economy_by_player[player_id] = {}
        else:
            self._objectives = [dict(item) for item in DEFAULT_OBJECTIVES]
            for index, player_id in enumerate(self._player_ids):
                template_player_id = DEFAULT_PLAYER_IDS[min(index, len(DEFAULT_PLAYER_IDS) - 1)]
                units = [deepcopy(unit) for unit in DEFAULT_UNIT_LAYOUT.get(template_player_id, ())]
                for unit in units:
                    unit["owner"] = player_id
                self._units_by_player[player_id] = units
                self._selection_by_player[player_id] = list(
                    DEFAULT_SELECTIONS.get(template_player_id, ())
                )
                self._production_by_player[player_id] = [
                    deepcopy(queue) for queue in DEFAULT_PRODUCTION.get(template_player_id, ())
                ]
                self._economy_by_player[player_id] = {
                    "credits": 1200 - index * 120,
                    "income_per_minute": 320 + index * 30,
                    "power": {
                        "produced": 100 + index * 15,
                        "used": 90 + index * 10,
                    },
                }
            self._apply_selection_flags()
        self._refresh_last_frame()

    def get_active_player(self) -> str:
        return self._player_ids[self._active_player_index % len(self._player_ids)]

    def observe(self, player: str) -> ArenaObservation:
        player_id = str(player or self.get_active_player())
        legal_actions = self._build_legal_actions(player_id)
        board_text = self._build_summary_text(player_id)
        return ArenaObservation(
            board_text=board_text,
            legal_moves=[str(item["id"]) for item in legal_actions],
            active_player=self.get_active_player(),
            last_move=self._last_move,
            metadata=self._build_metadata_payload(player_id),
            view={"text": board_text},
            context={
                "mode": "tick",
                "tick": self._tick,
                "step": self._step,
            },
        )

    def apply(self, action: ArenaAction) -> GameResult | None:
        if self._final_result is not None:
            return self._final_result

        active_player = self.get_active_player()
        resolved_action = self._resolve_action(action)
        if self._backend_mode == "native":
            return self._apply_native_action(action=action, resolved_action=resolved_action)
        legal_lookup = {
            str(candidate["id"]): candidate for candidate in self._build_legal_actions(active_player)
        }
        reward = 0.0
        if resolved_action.action_id not in legal_lookup or str(action.player) != active_player:
            self._illegal_move_count += 1
            self._last_move = f"illegal:{resolved_action.action_id}"
            reward = -0.25
        else:
            reward = self._apply_legal_action(
                active_player=active_player,
                resolved_action=resolved_action,
            )

        self._decision_count += 1
        self._tick += 1
        self._step += 1
        self._last_reward = reward
        self._advance_objectives()
        self._move_log.append(
            {
                "tick": self._tick,
                "step": self._step,
                "player_id": str(action.player),
                "action_id": resolved_action.action_id,
                "payload": dict(resolved_action.payload),
                "reward": reward,
                "last_move": self._last_move,
            }
        )

        if len(self._player_ids) > 1:
            self._active_player_index = (self._active_player_index + 1) % len(self._player_ids)

        if self._decision_count >= self._max_decisions:
            self._terminal = True
            self._final_result = self._build_result()

        self._refresh_last_frame()
        return self._final_result

    def get_last_frame(self) -> dict[str, Any]:
        if self._backend_mode == "native" and self._native_demo_script_mode == "live_stream":
            self._maybe_pump_native_demo_for_live_stream()
        if self._backend_mode == "native":
            self._refresh_last_frame()
        return dict(self._last_frame)

    def is_terminal(self) -> bool:
        return self._final_result is not None or self._terminal

    def _apply_legal_action(
        self,
        *,
        active_player: str,
        resolved_action: OpenRAResolvedAction,
    ) -> float:
        action_id = resolved_action.action_id
        payload = resolved_action.payload
        if action_id == "select_units":
            selected_ids = payload.get("unit_ids")
            if not isinstance(selected_ids, Sequence) or isinstance(selected_ids, (str, bytes)):
                selected_ids = self._selection_by_player.get(active_player, ())
            self._selection_by_player[active_player] = [str(item) for item in selected_ids][:3]
            self._apply_selection_flags()
            self._last_move = "select_units:" + ",".join(self._selection_by_player[active_player])
            return 0.05

        if action_id == "issue_command":
            target = payload.get("target") if isinstance(payload.get("target"), Mapping) else {}
            target_x = int(target.get("x", 18))
            target_y = int(target.get("y", 11))
            command = str(payload.get("command", "attack_move"))
            for unit in self._iter_selected_units(active_player):
                position = unit.get("position") if isinstance(unit.get("position"), Mapping) else {}
                unit["position"] = {
                    "x": max(int(position.get("x", 0)) + 1, target_x - 2),
                    "y": max(int(position.get("y", 0)) + 1, target_y - 2),
                }
                unit["status"] = "engaging"
            self._last_move = f"issue_command:{command}:{target_x},{target_y}"
            return 0.25

        if action_id == "queue_production":
            building_id = str(payload.get("building_id") or "barracks_1")
            unit_type = str(payload.get("unit_type") or "ranger")
            queue = self._ensure_primary_queue(active_player, building_id=building_id)
            queue["items"] = [
                *list(queue.get("items") or ()),
                {
                    "id": unit_type,
                    "label": unit_type.replace("_", " ").title(),
                    "progress": 0.15,
                },
            ]
            economy = self._economy_by_player[active_player]
            economy["credits"] = max(0, int(economy.get("credits", 0)) - 200)
            queue_label = str(queue["label"]).lower().replace(" ", "_")
            self._last_move = f"queue_production:{queue_label}:{unit_type}"
            return 0.2

        direction = str(payload.get("direction") or "left")
        distance = int(payload.get("distance", 6))
        self._last_move = f"camera_pan:{direction}:{distance}"
        return 0.01

    def _apply_native_action(
        self,
        *,
        action: ArenaAction,
        resolved_action: OpenRAResolvedAction,
    ) -> GameResult | None:
        active_player = self.get_active_player()
        if str(action.player) != active_player:
            self._illegal_move_count += 1
            self._last_move = f"illegal:{resolved_action.action_id}"
            self._last_reward = -0.25
        else:
            reward = 0.0
            if resolved_action.action_id == "bridge_input":
                if self._native_bridge is not None:
                    self._native_bridge.submit_input(dict(resolved_action.payload))
                event_type = str(resolved_action.payload.get("event_type") or "event")
                self._last_move = f"bridge_input:{event_type}"
            elif resolved_action.action_id == "noop":
                demo_payload = (
                    self._next_native_demo_payload()
                    if self._native_demo_script_mode == "noop"
                    else None
                )
                if demo_payload is not None and self._native_bridge is not None:
                    self._native_bridge.submit_input(demo_payload)
                    event_type = str(demo_payload.get("event_type") or "event")
                    self._last_move = f"demo_script:{self._native_demo_script}:{event_type}"
                else:
                    self._last_move = "noop"
            else:
                self._last_move = "noop"
            self._last_reward = reward

        self._wait_for_native_step_interval()
        self._decision_count += 1
        self._step += 1
        self._move_log.append(
            {
                "tick": self._tick,
                "step": self._step,
                "player_id": str(action.player),
                "action_id": resolved_action.action_id,
                "payload": dict(resolved_action.payload),
                "reward": self._last_reward,
                "last_move": self._last_move,
            }
        )
        if self._decision_count >= self._max_decisions:
            self._terminal = True
            self._final_result = self._build_result()
        self._refresh_last_frame()
        return self._final_result

    def _next_native_demo_payload(self) -> dict[str, Any] | None:
        if self._native_demo_script == "camera_tour":
            payload = CAMERA_TOUR_NATIVE_EVENTS[self._native_demo_step % len(CAMERA_TOUR_NATIVE_EVENTS)]
            self._native_demo_step += 1
            if payload is None:
                return None
            return dict(payload)
        if self._native_demo_script == "unit_patrol":
            payload = UNIT_PATROL_NATIVE_EVENTS[self._native_demo_step % len(UNIT_PATROL_NATIVE_EVENTS)]
            self._native_demo_step += 1
            if payload is None:
                return None
            return self._resolve_native_demo_payload(payload)
        return None

    def _maybe_pump_native_demo_for_live_stream(self) -> None:
        if self._backend_mode != "native" or self._native_bridge is None:
            return
        if self._native_demo_script is None:
            return
        now = time.monotonic()
        if (
            self._native_last_demo_pump_monotonic > 0.0
            and now - self._native_last_demo_pump_monotonic < self._native_step_interval_s
        ):
            return
        self._native_last_demo_pump_monotonic = now
        payload = self._next_native_demo_payload()
        if payload is None:
            self._last_move = "noop"
            return
        self._native_bridge.submit_input(payload)
        event_type = str(payload.get("event_type") or "event")
        self._last_move = f"demo_stream:{self._native_demo_script}:{event_type}"

    def _resolve_native_demo_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        event_type = str(payload.get("event_type") or "mouse_move")
        modifiers = _normalize_string_sequence(payload.get("modifiers"))
        if event_type.startswith("key"):
            return {
                "event_type": event_type,
                "key": str(payload.get("key") or ""),
                "modifiers": list(modifiers),
            }

        viewport_source = self._native_state.get("viewport")
        viewport = (
            {
                "width": int(viewport_source.get("width", self._viewport["width"])),
                "height": int(viewport_source.get("height", self._viewport["height"])),
            }
            if isinstance(viewport_source, Mapping)
            else dict(self._viewport)
        )
        x = _scale_native_demo_coordinate(int(payload.get("x", 0)), axis="x", viewport=viewport)
        y = _scale_native_demo_coordinate(int(payload.get("y", 0)), axis="y", viewport=viewport)
        button = str(payload.get("button") or ("none" if event_type == "mouse_scroll" else "left"))
        resolved_payload = {
            "event_type": event_type,
            "button": button,
            "buttons": [] if button == "none" else [button],
            "x": x,
            "y": y,
            "viewport": viewport,
            "modifiers": list(modifiers),
        }
        if event_type == "mouse_scroll":
            resolved_payload["delta_x"] = int(payload.get("delta_x", 0))
            resolved_payload["delta_y"] = int(payload.get("delta_y", 0))
        return resolved_payload

    def _iter_selected_units(self, player_id: str) -> list[dict[str, Any]]:
        selected_ids = set(self._selection_by_player.get(player_id, ()))
        return [
            unit
            for unit in self._units_by_player.get(player_id, [])
            if str(unit.get("id")) in selected_ids
        ]

    def _apply_selection_flags(self) -> None:
        for player_id, units in self._units_by_player.items():
            selected_ids = set(self._selection_by_player.get(player_id, ()))
            for unit in units:
                unit["selected"] = str(unit.get("id")) in selected_ids

    def _ensure_primary_queue(self, player_id: str, *, building_id: str) -> dict[str, Any]:
        queues = self._production_by_player.setdefault(player_id, [])
        for queue in queues:
            if str(queue.get("building_id")) == building_id:
                return queue
        created = {
            "building_id": building_id,
            "label": building_id.replace("_", " ").title(),
            "items": [],
        }
        queues.append(created)
        return created

    def _advance_objectives(self) -> None:
        if self._decision_count >= 3:
            self._objectives[0]["status"] = "complete"
        if self._decision_count >= 4:
            self._objectives[1]["status"] = "active"
        if self._decision_count >= self._max_decisions:
            self._objectives[1]["status"] = "complete"

        for player_id, queues in self._production_by_player.items():
            for queue in queues:
                advanced_items: list[dict[str, Any]] = []
                for item in list(queue.get("items") or ()):
                    progress = min(1.0, float(item.get("progress", 0.0)) + 0.18)
                    advanced_item = dict(item)
                    advanced_item["progress"] = progress
                    advanced_items.append(advanced_item)
                queue["items"] = advanced_items

            economy = self._economy_by_player[player_id]
            economy["credits"] = int(economy.get("credits", 0)) + int(
                economy.get("income_per_minute", 0) / 8
            )

    def _sync_native_projection_state(self) -> None:
        if self._backend_mode != "native" or not self._player_ids:
            return
        active_player = self.get_active_player()
        unit_count = max(0, _coerce_optional_int(self._native_state.get("local_player_actor_count")) or 0)
        if unit_count <= 0:
            self._units_by_player[active_player] = []
            self._selection_by_player[active_player] = []
            return

        origin_x, origin_y, clamp_bounds = self._resolve_native_unit_anchor()
        offsets = (
            (0, 0),
            (1, 1),
            (-1, 1),
            (2, 0),
            (-2, 0),
            (0, 2),
            (1, -1),
            (-1, -1),
        )
        units: list[dict[str, Any]] = []
        for index in range(unit_count):
            dx, dy = offsets[index % len(offsets)]
            position_x = origin_x + dx
            position_y = origin_y + dy
            if clamp_bounds is not None:
                min_x, max_x, min_y, max_y = clamp_bounds
                position_x = max(min_x, min(position_x, max_x))
                position_y = max(min_y, min(position_y, max_y))
            unit_id = f"native_unit_{index + 1}"
            units.append(
                {
                    "id": unit_id,
                    "owner": active_player,
                    "label": f"Local Unit {index + 1}",
                    "kind": "vehicle" if index == 0 else "infantry",
                    "hp": max(40, 100 - index * 8),
                    "status": "ready" if bool(self._native_state.get("is_ready")) else "spawning",
                    "position": {
                        "x": position_x,
                        "y": position_y,
                    },
                    "selected": index == 0,
                }
            )

        self._units_by_player[active_player] = units
        self._selection_by_player[active_player] = [str(units[0]["id"])] if units else []
        self._apply_selection_flags()

    def _resolve_native_unit_anchor(self) -> tuple[int, int, tuple[int, int, int, int] | None]:
        map_size = self._reference_preview.map_size if self._reference_preview is not None else None
        bounds = self._reference_preview.bounds if self._reference_preview is not None else None
        min_x = 0
        min_y = 0
        max_x = 63
        max_y = 63
        if isinstance(map_size, Mapping):
            width = max(1, int(map_size.get("width", 64)))
            height = max(1, int(map_size.get("height", 64)))
            max_x = width - 1
            max_y = height - 1
        if isinstance(bounds, Mapping):
            min_x = int(bounds.get("x", min_x))
            min_y = int(bounds.get("y", min_y))
            max_x = max(min_x, min_x + int(bounds.get("width", max_x - min_x + 1)) - 1)
            max_y = max(min_y, min_y + int(bounds.get("height", max_y - min_y + 1)) - 1)
        fallback_x = min_x + max(0, (max_x - min_x) // 2)
        fallback_y = min_y + max(0, (max_y - min_y) // 2)

        camera_center = self._native_state.get("camera_center")
        if isinstance(camera_center, Mapping):
            origin_x = _coerce_optional_int(camera_center.get("x"))
            origin_y = _coerce_optional_int(camera_center.get("y"))
        else:
            origin_x = None
            origin_y = None

        def _normalize_coordinate(value: int | None, *, fallback: int, max_span: int) -> int:
            if value is None:
                return fallback
            normalized = int(value)
            if abs(normalized) > max(max_span * 4, 256):
                normalized = int(round(normalized / 1024.0))
            return normalized

        resolved_x = _normalize_coordinate(origin_x, fallback=fallback_x, max_span=max_x - min_x + 1)
        resolved_y = _normalize_coordinate(origin_y, fallback=fallback_y, max_span=max_y - min_y + 1)
        resolved_x = max(min_x, min(resolved_x, max_x))
        resolved_y = max(min_y, min(resolved_y, max_y))
        return resolved_x, resolved_y, (min_x, max_x, min_y, max_y)

    def _build_legal_actions(self, player_id: str) -> list[dict[str, Any]]:
        if self._backend_mode == "native":
            return [
                {
                    "id": "noop",
                    "label": "No-op",
                    "text": "No-op",
                    "payloadSchema": {},
                },
                {
                    "id": "bridge_input",
                    "label": "Native input",
                    "text": "Native input",
                    "payloadSchema": {
                        "event_type": "mouse_down",
                        "x": 0,
                        "y": 0,
                        "button": "left",
                    },
                },
            ]
        selection = list(self._selection_by_player.get(player_id, ()))
        if not selection:
            selection = [
                str(unit.get("id"))
                for unit in self._units_by_player.get(player_id, [])[:2]
            ]
        command_target = COMMAND_TARGETS[self._decision_count % len(COMMAND_TARGETS)]
        pan_direction = PAN_DIRECTIONS[self._decision_count % len(PAN_DIRECTIONS)]
        queue_choice = QUEUE_ROTATION[self._decision_count % len(QUEUE_ROTATION)]
        return [
            {
                "id": "select_units",
                "label": "Select units",
                "text": "Select units",
                "payloadSchema": {"unit_ids": selection},
            },
            {
                "id": "issue_command",
                "label": "Issue command",
                "text": "Issue command",
                "payloadSchema": {
                    "command": "attack_move",
                    "target": dict(command_target),
                },
            },
            {
                "id": "queue_production",
                "label": "Queue production",
                "text": "Queue production",
                "payloadSchema": {
                    "building_id": "barracks_1",
                    "unit_type": queue_choice["id"],
                },
            },
            {
                "id": "camera_pan",
                "label": "Camera pan",
                "text": "Camera pan",
                "payloadSchema": {
                    "direction": pan_direction,
                    "distance": 6,
                },
            },
        ]

    def _build_summary_text(self, player_id: str) -> str:
        if self._backend_mode == "native":
            del player_id
            return (
                "Native OpenRA runtime. Left click to select, right click to command, "
                "mouse wheel to zoom, arrow or WASD to pan."
            )
        economy = self._economy_by_player.get(player_id, {})
        credits = int(economy.get("credits", 0))
        power = economy.get("power") if isinstance(economy.get("power"), Mapping) else {}
        power_delta = int(power.get("produced", 0)) - int(power.get("used", 0))
        primary_objective = self._objectives[0]["label"]
        return f"Credits {credits} | Power {power_delta:+d} | Objective {primary_objective}"

    def _build_frame_image_data_url(self) -> str:
        units = [
            unit
            for player_units in self._units_by_player.values()
            for unit in player_units
        ]
        selected = set(self._selection_by_player.get(self.get_active_player(), ()))
        circles: list[str] = []
        for unit in units:
            position = unit.get("position") if isinstance(unit.get("position"), Mapping) else {}
            x = 90 + int(position.get("x", 0)) * 18
            y = 80 + int(position.get("y", 0)) * 18
            owner = str(unit.get("owner") or "")
            fill = "#2a7f62" if owner == self.get_active_player() else "#8b4b3d"
            stroke = "#f7f4e8" if str(unit.get("id")) in selected else "#183a2f"
            circles.append(
                f"<circle cx='{x}' cy='{y}' r='14' fill='{fill}' stroke='{stroke}' stroke-width='3' />"
            )
        objective_text = escape(str(self._objectives[0]["label"]))
        env_label = escape(self._env_id)
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1280 720'>"
            "<defs><linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>"
            "<stop offset='0%' stop-color='#193a33'/><stop offset='100%' stop-color='#314d2f'/>"
            "</linearGradient></defs>"
            "<rect width='1280' height='720' fill='url(#bg)'/>"
            "<rect x='42' y='42' width='1196' height='636' rx='28' fill='rgba(240,233,209,0.08)' "
            "stroke='rgba(240,233,209,0.32)' stroke-width='4'/>"
            "<path d='M120 560 C320 460 480 520 690 430 S980 320 1160 380' "
            "stroke='rgba(238,214,160,0.45)' stroke-width='18' fill='none' stroke-linecap='round'/>"
            "<rect x='76' y='76' width='340' height='88' rx='18' fill='rgba(16,28,24,0.52)'/>"
            f"<text x='100' y='126' font-size='34' fill='#f7f4e8'>OpenRA {env_label}</text>"
            f"<text x='100' y='550' font-size='28' fill='#f7f4e8'>Objective: {objective_text}</text>"
            f"{''.join(circles)}"
            "</svg>"
        )
        encoded_svg = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{encoded_svg}"

    def _build_frame_image_payload(self) -> dict[str, Any]:
        if self._backend_mode == "native":
            data_url = self._native_state.get("frame_data_url")
            if isinstance(data_url, str) and data_url.startswith("data:"):
                return {
                    "data_url": data_url,
                    "mimeType": str(self._native_state.get("mime_type") or "image/png"),
                }
        if self._reference_preview is not None:
            return {
                "data_url": self._reference_preview.data_url,
                "mimeType": self._reference_preview.mime_type,
            }
        return {
            "data_url": self._build_frame_image_data_url(),
            "mimeType": "image/svg+xml",
        }

    def _build_map_payload(self) -> dict[str, Any]:
        if self._backend_mode == "native":
            viewport = self._native_state.get("viewport")
            image_size = (
                {
                    "width": int(viewport.get("width", self._viewport["width"])),
                    "height": int(viewport.get("height", self._viewport["height"])),
                }
                if isinstance(viewport, Mapping)
                else dict(self._viewport)
            )
            return {
                "id": self._env_id,
                "mod_id": self._mod_id,
                "title": self._reference_preview.title if self._reference_preview is not None else self._env_id,
                "map_size": self._reference_preview.map_size if self._reference_preview is not None else None,
                "bounds": self._reference_preview.bounds if self._reference_preview is not None else None,
                "image_size": image_size,
                "preview_source": "native_runtime",
            }
        return {
            "id": self._env_id,
            "mod_id": self._mod_id,
            "title": self._reference_preview.title if self._reference_preview is not None else self._env_id,
            "map_size": self._reference_preview.map_size if self._reference_preview is not None else None,
            "bounds": self._reference_preview.bounds if self._reference_preview is not None else None,
            "image_size": self._reference_preview.image_size if self._reference_preview is not None else None,
            "preview_source": (
                self._reference_preview.preview_source
                if self._reference_preview is not None
                else "generated_stub"
            ),
        }

    def _build_metadata_payload(self, player_id: str) -> dict[str, Any]:
        selection = list(self._selection_by_player.get(player_id, ()))
        primary_unit_id = selection[0] if selection else None
        economy = deepcopy(self._economy_by_player.get(player_id, {}))
        units = [deepcopy(item) for item in self._units_by_player.get(player_id, ())]
        production = [deepcopy(item) for item in self._production_by_player.get(player_id, ())]
        return {
            "env_id": self._env_id,
            "map_id": self._env_id,
            "player_id": player_id,
            "backend_mode": self._backend_mode,
            "stream_id": self._stream_id,
            "map": self._build_map_payload(),
            "economy": economy,
            "objectives": [dict(item) for item in self._objectives],
            "selection": {
                "unit_ids": selection,
                "primary_unit_id": primary_unit_id,
            },
            "units": units,
            "production": {"queues": production},
            "reward": self._last_reward,
            "last_move": self._last_move,
        }

    def _build_frame_payload(self) -> dict[str, Any]:
        active_player = self.get_active_player()
        board_text = self._build_summary_text(active_player)
        image_payload = self._build_frame_image_payload()
        native_tick = _coerce_optional_int(self._native_state.get("engine_tick")) if self._backend_mode == "native" else None
        if native_tick is not None:
            self._tick = native_tick
        return {
            "env_id": self._env_id,
            "tick": self._tick,
            "step": self._step,
            "move_count": self._decision_count,
            "last_move": self._last_move,
            "reward": self._last_reward,
            "active_player_id": active_player,
            "stream_id": self._stream_id,
            "viewport": dict(self._viewport),
            "board_text": board_text,
            "metadata": self._build_metadata_payload(active_player),
            "legal_actions": {
                "items": self._build_legal_actions(active_player),
            },
            "view": {
                "text": board_text,
                "image": image_payload,
            },
            "media": {
                "primary": {
                    "mediaId": f"openra-frame-{self._tick}",
                    "transport": "http_pull",
                    "mimeType": str(image_payload["mimeType"]),
                    "url": str(image_payload["data_url"]),
                }
            },
        }

    def _refresh_last_frame(self) -> None:
        self._poll_native_bridge_state()
        self._last_frame = self._build_frame_payload()

    def _build_result(self) -> GameResult:
        winner = self._player_ids[0] if self._player_ids else None
        final_board = self._build_summary_text(self._player_ids[0])
        return GameResult(
            winner=winner,
            result="mission_complete",
            reason=(
                "native_runtime_step_limit"
                if self._backend_mode == "native"
                else "stub_objectives_complete"
            ),
            move_count=self._decision_count,
            illegal_move_count=self._illegal_move_count,
            final_board=final_board,
            move_log=tuple(self._move_log),
            rule_profile=(
                f"openra_native::{self._env_id}"
                if self._backend_mode == "native"
                else f"openra_stub::{self._env_id}"
            ),
        )

    def close(self) -> None:
        if self._native_bridge is not None:
            closer = getattr(self._native_bridge, "close", None)
            if callable(closer):
                closer()

    def _wait_for_native_step_interval(self) -> None:
        if self._backend_mode != "native":
            return
        now = time.monotonic()
        elapsed = now - self._native_last_apply_monotonic
        remaining = self._native_step_interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._native_last_apply_monotonic = time.monotonic()

    def _poll_native_bridge_state(self) -> None:
        if self._backend_mode != "native" or self._native_bridge is None:
            return
        state = self._native_bridge.read_state()
        if isinstance(state, Mapping):
            self._native_state = dict(state)
            native_tick = _coerce_optional_int(self._native_state.get("engine_tick"))
            if native_tick is not None:
                self._tick = native_tick
            self._sync_native_projection_state()

    @staticmethod
    def _resolve_action(action: ArenaAction) -> OpenRAResolvedAction:
        raw_text = str(action.raw or action.move or "").strip()
        action_id = str(action.move or "").strip()
        payload: dict[str, Any] = {}
        if raw_text.startswith("{"):
            try:
                decoded = json.loads(raw_text)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, Mapping):
                decoded_action_id = (
                    decoded.get("action_id")
                    or decoded.get("actionId")
                    or decoded.get("action")
                    or decoded.get("move")
                    or decoded.get("id")
                )
                if decoded_action_id is not None:
                    action_id = str(decoded_action_id).strip()
                decoded_payload = decoded.get("payload")
                if isinstance(decoded_payload, Mapping):
                    payload = dict(decoded_payload)
                else:
                    payload = {
                        key: value
                        for key, value in decoded.items()
                        if key not in {"action_id", "actionId", "action", "move", "id"}
                    }
        return OpenRAResolvedAction(
            action_id=action_id or raw_text,
            payload=payload,
            raw_text=raw_text,
        )


__all__ = ["OpenRAArenaEnvironment", "OpenRAResolvedAction"]


def _coerce_optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
