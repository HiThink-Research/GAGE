"""PettingZoo ws_rgb replay adapter based on sample artifacts."""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gage_eval.role.arena.games.pettingzoo.env import PettingZooAecArenaEnvironment
from gage_eval.role.arena.types import ArenaAction

_TASK_PREFIX = "pettingzoo_"
_TASK_SUFFIXES = (
    "_ws_rgb",
    "_dummy",
    "_ai",
    "_litellm",
    "_local",
)
_GAME_VERSIONS: dict[str, str] = {
    "basketball_pong": "v3",
    "boxing": "v2",
    "combat_plane": "v2",
    "combat_tank": "v2",
    "double_dunk": "v3",
    "entombed_competitive": "v3",
    "entombed_cooperative": "v3",
    "flag_capture": "v2",
    "foozpong": "v3",
    "ice_hockey": "v2",
    "joust": "v3",
    "mario_bros": "v3",
    "maze_craze": "v3",
    "othello": "v3",
    "pong": "v3",
    "space_invaders": "v2",
    "space_war": "v2",
    "surround": "v2",
    "tennis": "v3",
    "video_checkers": "v4",
    "volleyball_pong": "v3",
    "wizard_of_wor": "v3",
}


class ReplayFrameCursor:
    """Resolve replay frame by elapsed wall-clock time."""

    def __init__(self, frames: Sequence[Mapping[str, Any]], *, fps: float) -> None:
        self._frames = [dict(item) for item in frames if isinstance(item, Mapping)]
        if not self._frames:
            self._frames = [
                {
                    "board_text": "No replay frames available.",
                    "metadata": {"replay_total": 0, "replay_index": 0},
                }
            ]
        self._fps = max(0.1, float(fps))
        self._frame_interval_s = 1.0 / self._fps
        self._started_at: Optional[float] = None
        self._last_advance_at: Optional[float] = None
        self._index = 0

    def frame_source(self) -> dict[str, Any]:
        """Return one frame payload for ws_rgb polling."""

        # STEP 1: Start replay cursor at first frame pull.
        now = time.monotonic()
        if self._started_at is None:
            self._started_at = now
            self._last_advance_at = now
        elif self._index < len(self._frames) - 1:
            last_advance_at = self._last_advance_at if self._last_advance_at is not None else now
            elapsed_since_advance = max(0.0, now - last_advance_at)
            # STEP 2: Advance at most one frame per pull to avoid skipping.
            if elapsed_since_advance >= self._frame_interval_s:
                self._index += 1
                self._last_advance_at = now

        started_at = self._started_at if self._started_at is not None else now
        elapsed_s = max(0.0, now - started_at)
        payload = dict(self._frames[self._index])
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = dict(metadata)
        metadata["replay_elapsed_s"] = elapsed_s
        metadata["replay_index"] = self._index
        metadata["replay_total"] = len(self._frames)
        payload["metadata"] = metadata
        return payload


def build_ws_rgb_replay_display(
    sample_record: Mapping[str, Any],
    *,
    task_id: str,
    fps: float,
    max_frames: int,
) -> dict[str, Any]:
    """Build ws_rgb display payload for one PettingZoo sample record.

    Args:
        sample_record: Top-level sample JSON payload.
        task_id: Task id from sample record.
        fps: Replay fps for frame cursor.
        max_frames: Optional upper bound of replay frames (0 means no cap).

    Returns:
        A dict containing ``display_id``, ``label``, ``human_player_id``, and
        ``frame_source`` callable used by WsRgbHubServer.
    """

    sample = sample_record.get("sample")
    if not isinstance(sample, Mapping):
        raise ValueError("sample_json_missing_sample")
    sample_id = str(sample.get("id") or "sample")
    env_id = _infer_env_id(sample, task_id)
    frames = _build_replay_frames(
        sample,
        env_id=env_id,
        max_frames=max(0, int(max_frames)),
    )
    cursor = ReplayFrameCursor(frames, fps=float(fps))

    human_player_id = "player_0"
    metadata = sample.get("metadata")
    if isinstance(metadata, Mapping):
        player_ids = metadata.get("player_ids")
        if isinstance(player_ids, Sequence) and not isinstance(player_ids, (str, bytes)) and player_ids:
            human_player_id = str(player_ids[0])

    return {
        "display_id": f"replay:{sample_id}:pettingzoo_ws_rgb",
        "label": f"pettingzoo_replay:{env_id}",
        "human_player_id": human_player_id,
        "frame_source": cursor.frame_source,
    }


def _normalize_task_game_name(task_id: str) -> str:
    text = str(task_id or "").strip().lower()
    if text.startswith(_TASK_PREFIX):
        text = text[len(_TASK_PREFIX) :]
    changed = True
    while changed and text:
        changed = False
        for suffix in _TASK_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                changed = True
    return text.strip("_")


def _can_import_env(env_id: str) -> bool:
    if not env_id:
        return False
    if ":" in env_id:
        module_name, attr = env_id.split(":", 1)
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return False
        return hasattr(module, attr)
    try:
        module = importlib.import_module(env_id)
        return hasattr(module, "env") or hasattr(module, "parallel_env")
    except Exception:
        pass
    if "." not in env_id:
        return False
    module_name, attr = env_id.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    return hasattr(module, attr)


def _load_replay_meta(sample: Mapping[str, Any]) -> dict[str, Any]:
    predict_result = sample.get("predict_result")
    if not isinstance(predict_result, list):
        return {}
    for item in predict_result:
        if not isinstance(item, Mapping):
            continue
        replay_path = item.get("replay_path")
        if not replay_path:
            continue
        path = Path(str(replay_path)).expanduser()
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, Mapping):
            meta = payload.get("meta")
            if isinstance(meta, Mapping):
                return dict(meta)
    return {}


def _infer_env_id(sample: Mapping[str, Any], task_id: str) -> str:
    metadata = sample.get("metadata")
    candidate_meta = metadata if isinstance(metadata, Mapping) else {}
    replay_meta = _load_replay_meta(sample)
    candidates: list[str] = []

    # STEP 1: Try explicit replay/meta env ids first.
    for key in ("env_id",):
        value = replay_meta.get(key)
        if value:
            candidates.append(str(value))
    for key in ("env_id", "pettingzoo_env_id"):
        value = candidate_meta.get(key)
        if value:
            candidates.append(str(value))

    # STEP 2: Infer env id from task id (prefer this over metadata.game_id).
    game_name = _normalize_task_game_name(task_id)
    if game_name:
        version = _GAME_VERSIONS.get(game_name, "v3")
        candidates.extend(
            [
                f"pettingzoo.atari.{game_name}_{version}",
                f"pettingzoo.atari.{game_name}_v3",
                f"pettingzoo.atari.{game_name}_v2",
                f"pettingzoo.atari.{game_name}_v1",
            ]
        )

    # STEP 3: Keep metadata.game_id as fallback because it can be stale in datasets.
    game_id = candidate_meta.get("game_id")
    if game_id:
        candidates.append(str(game_id))

    # STEP 4: Return the first importable candidate.
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if _can_import_env(normalized):
            return normalized

    raise ValueError(
        "pettingzoo_env_id_unresolved; provide metadata.env_id or use task_id that maps to known atari env"
    )


def _resolve_game_log(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    predict_result = sample.get("predict_result")
    if not isinstance(predict_result, list):
        return []
    for item in predict_result:
        if not isinstance(item, Mapping):
            continue
        game_log = item.get("game_log")
        if isinstance(game_log, list):
            return [dict(entry) for entry in game_log if isinstance(entry, Mapping)]
    return []


def _build_replay_frames(
    sample: Mapping[str, Any],
    *,
    env_id: str,
    max_frames: int,
) -> list[dict[str, Any]]:
    game_log = _resolve_game_log(sample)
    if not game_log:
        raise ValueError("sample_game_log_missing")

    metadata = sample.get("metadata")
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    player_ids_raw = metadata_map.get("player_ids")
    player_ids: Optional[list[str]] = None
    if isinstance(player_ids_raw, Sequence) and not isinstance(player_ids_raw, (str, bytes)):
        player_ids = [str(item) for item in player_ids_raw if item is not None]

    env_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
    max_cycles = len(game_log) + 16
    if max_cycles > 0:
        env_kwargs["max_cycles"] = max_cycles

    arena_env = PettingZooAecArenaEnvironment(
        env_id=env_id,
        env_kwargs=env_kwargs,
        player_ids=player_ids,
        include_raw_obs=False,
        use_action_meanings=False,
        illegal_policy={"retry": 0, "on_fail": "loss"},
    )

    # STEP 1: Capture initial frame from reset state.
    frames: list[dict[str, Any]] = [dict(arena_env.get_last_frame())]

    # STEP 2: Replay each move from game_log and append frame snapshots.
    for entry in game_log:
        active_player = arena_env.get_active_player()
        arena_env.observe(active_player)
        frames.append(dict(arena_env.get_last_frame()))

        action_id = entry.get("action_id")
        move = entry.get("move")
        if action_id is not None:
            move_text = str(action_id)
        elif move is not None:
            move_text = str(move)
        else:
            continue

        result = arena_env.apply(
            ArenaAction(
                player=str(active_player),
                move=move_text,
                raw=move_text,
                metadata={"source": "post_run_replay"},
            )
        )
        frames.append(dict(arena_env.get_last_frame()))
        if result is not None:
            break
        if max_frames > 0 and len(frames) >= max_frames:
            break

    return frames
