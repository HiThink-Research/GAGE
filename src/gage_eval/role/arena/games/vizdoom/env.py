"""ViZDoom arena environment adapter."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import base64
import importlib
import io
from pathlib import Path
import time
from typing import Any, Dict, Optional, Sequence

from gage_eval.registry import registry
from gage_eval.role.arena.replay_paths import resolve_replay_manifest_path
from gage_eval.role.arena.games.vizdoom.observation import ViZDoomPromptBuilder
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

DEFAULT_ACTION_LABELS = ("turn_left", "turn_right", "move_forward", "move_backward", "attack")
ACTION_ID_MAPPING = {
    1: "ATTACK",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}
ALLOWED_ACTION_IDS = (1, 2, 3)


@dataclass
class ViZDoomEnvConfig:
    """Configuration for the ViZDoom arena environment."""

    use_single_process: bool = False
    render_mode: Optional[str] = None
    pov_view: Optional[str] = None
    show_automap: bool = True
    automap_scale: int = 3
    automap_follow: bool = False
    automap_stride: int = 1
    show_pov: bool = True
    capture_pov: bool = False
    pov_stride: int = 2
    allow_respawn: bool = False
    respawn_grace_steps: int = 0
    no_attack_seconds: float = 10.0
    max_steps: int = 4000
    action_repeat: int = 4
    sleep_s: float = 0.02
    port: Optional[int] = None
    config_path: Optional[str] = None
    replay_output_dir: Optional[str] = None
    game_id: str = "vizdoom_multi_duel"
    tick_rate_hz: Optional[float] = None
    frame_stride: int = 1
    time_source: str = "wall_clock"
    obs_image: bool = False
    obs_image_history_len: int = 1
    replay_in_env: bool = True
    action_labels: Optional[Sequence[str]] = None
    allow_partial_actions: bool = False
    reset_retry_count: int = 3
    death_check_warmup_steps: int = 8


@registry.asset(
    "arena_impls",
    "vizdoom_env_v1",
    desc="ViZDoom arena environment adapter (demo integration)",
    tags=("vizdoom", "arena"),
)
class ViZDoomArenaEnvironment:
    """Arena environment that wraps the ViZDoom multiplayer demo env."""

    def __init__(
        self,
        *,
        player_ids: Optional[Sequence[str]] = None,
        player_names: Optional[Dict[str, str]] = None,
        start_player_id: Optional[str] = None,
        use_single_process: bool = False,
        render_mode: Optional[str] = None,
        pov_view: Optional[str] = None,
        show_automap: bool = True,
        automap_scale: int = 3,
        automap_follow: bool = False,
        automap_stride: int = 1,
        show_pov: bool = True,
        capture_pov: bool = False,
        pov_stride: int = 2,
        allow_respawn: bool = False,
        respawn_grace_steps: int = 0,
        no_attack_seconds: float = 10.0,
        max_steps: int = 4000,
        action_repeat: int = 4,
        sleep_s: float = 0.02,
        port: Optional[int] = None,
        config_path: Optional[str] = None,
        replay_output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        game_id: str = "vizdoom_multi_duel",
        tick_rate_hz: Optional[float] = None,
        frame_stride: int = 1,
        time_source: str = "wall_clock",
        obs_image: bool = False,
        obs_image_history_len: int = 1,
        replay_in_env: bool = True,
        action_labels: Optional[Sequence[str]] = None,
        allow_partial_actions: bool = False,
        reset_retry_count: int = 3,
        death_check_warmup_steps: int = 8,
        **_: object,
    ) -> None:
        """Initialize the ViZDoom arena environment adapter.

        Args:
            player_ids: Ordered player identifiers.
            player_names: Optional display name mapping.
            start_player_id: Optional starting player id.
            use_single_process: Whether to run ViZDoom in single-process mode.
            render_mode: Optional render mode string for the backend.
            show_automap: Whether to render automap frames.
            automap_scale: Automap scaling factor.
            automap_follow: Whether the automap camera follows the player.
            automap_stride: Frame stride for automap capture.
            show_pov: Whether to render POV frames.
            capture_pov: Whether to capture POV frames without opening POV window.
            pov_stride: Frame stride for POV capture.
            allow_respawn: Whether to allow respawn in the backend.
            respawn_grace_steps: Grace period before respawn.
            no_attack_seconds: Seconds before allowing attacks.
            max_steps: Maximum steps before termination.
            action_repeat: Backend action repeat.
            sleep_s: Sleep interval between ticks.
            port: Optional backend port.
            config_path: Path to the ViZDoom config file.
            replay_output_dir: Output directory for replay files.
            game_id: Replay game id label.
            tick_rate_hz: Optional tick rate metadata.
            frame_stride: Frame sampling stride for replays.
            time_source: Timestamp source label.
            obs_image: Whether to include image frames in observations.
            obs_image_history_len: Number of recent POV frames to expose in observations.
            replay_in_env: Whether the environment writes replays directly.
            action_labels: Optional action label list for legal actions.
        """

        self._cfg = ViZDoomEnvConfig(
            use_single_process=bool(use_single_process),
            render_mode=render_mode,
            pov_view=pov_view,
            show_automap=bool(show_automap),
            automap_scale=int(automap_scale),
            automap_follow=bool(automap_follow),
            automap_stride=int(automap_stride),
            show_pov=bool(show_pov),
            capture_pov=bool(capture_pov),
            pov_stride=int(pov_stride),
            allow_respawn=bool(allow_respawn),
            respawn_grace_steps=int(respawn_grace_steps),
            no_attack_seconds=float(no_attack_seconds),
            max_steps=int(max_steps),
            action_repeat=int(action_repeat),
            sleep_s=float(sleep_s),
            port=port,
            config_path=config_path,
            replay_output_dir=replay_output_dir,
            game_id=str(game_id),
            tick_rate_hz=tick_rate_hz,
            frame_stride=int(frame_stride),
            time_source=str(time_source),
            obs_image=bool(obs_image),
            obs_image_history_len=max(1, int(obs_image_history_len)),
            replay_in_env=bool(replay_in_env),
            action_labels=tuple(action_labels) if action_labels else None,
            allow_partial_actions=bool(allow_partial_actions),
            reset_retry_count=max(1, int(reset_retry_count)),
            death_check_warmup_steps=max(0, int(death_check_warmup_steps)),
        )
        self._env = self._build_env(self._cfg)
        self._player_ids = [str(pid) for pid in (player_ids or ["p0", "p1"])]
        self._player_names = dict(player_names or {})
        self._start_player_id = str(start_player_id or self._player_ids[0])
        self._player_index = {pid: idx for idx, pid in enumerate(self._player_ids)}
        self._player_id_by_index = {idx: pid for pid, idx in self._player_index.items()}
        self._run_id = str(run_id) if run_id else None
        self._sample_id = str(sample_id) if sample_id else None
        self._action_label_to_id = (
            {str(label): idx for idx, label in enumerate(self._cfg.action_labels)}
            if self._cfg.action_labels
            else {}
        )
        self._obs_by_player: Dict[int, Dict[str, Any]] = {}
        self._last_rewards: Dict[int, float] = {}
        self._last_info: Dict[str, Any] = {}
        self._done = False
        self._tick = 0
        self._pending_actions: Dict[int, ArenaAction] = {}
        self._move_log: list[Dict[str, Any]] = []
        self._replay = _ReplayWriter(
            self._cfg.replay_output_dir,
            run_id=self._run_id,
            sample_id=self._sample_id,
        )
        self._start_ts = time.time()
        self._active_idx = 0
        self._last_frame_payload: Dict[str, Any] = {}
        self._display_pov_view = self._normalize_pov_view(self._cfg.pov_view)
        self._obs_image_history: Dict[int, deque[Dict[str, Any]]] = {
            idx: deque(maxlen=max(1, int(self._cfg.obs_image_history_len)))
            for idx in self._player_index.values()
        }
        self._prompt_builder = ViZDoomPromptBuilder()
        self._reported_session_tick = 0

    def reset(self) -> None:
        """Reset the environment to its initial state."""

        # STEP 1: Reset the underlying ViZDoom environment.
        self._obs_by_player = self._env.reset(seed=None)
        self._last_rewards = {idx: 0.0 for idx in self._player_index.values()}
        self._last_info = {}
        self._done = False
        self._tick = 0
        self._pending_actions = {}
        self._move_log = []
        self._start_ts = time.time()
        self._active_idx = 0
        self._last_frame_payload = {}
        self._display_pov_view = self._normalize_pov_view(self._cfg.pov_view)
        self._obs_image_history = {
            idx: deque(maxlen=max(1, int(self._cfg.obs_image_history_len)))
            for idx in self._player_index.values()
        }
        self._reported_session_tick = 0

        # STEP 2: Initialize replay if enabled.
        if self._cfg.replay_in_env:
            meta = ReplayMeta(
                game_id=self._cfg.game_id,
                players=list(self._player_index.values()),
                tick_mode=True,
                tick_rate_hz=self._cfg.tick_rate_hz,
                frame_stride=self._cfg.frame_stride or None,
                time_source=self._cfg.time_source,
            )
            self._replay.start(meta)
        self._refresh_observation_image_history()
        self._refresh_last_frame_payload()

    def get_active_player(self) -> str:
        """Return the player_id of the participant who should act next."""

        if not self._player_ids:
            return "p0"
        player_id = self._player_ids[self._active_idx % len(self._player_ids)]
        self._active_idx = (self._active_idx + 1) % len(self._player_ids)
        return player_id

    def observe(self, player: str) -> ArenaObservation:
        """Return an observation for the given player_id."""

        player_id = str(player)
        idx = self._resolve_player_index(player_id)
        raw_obs = self._obs_by_player.get(idx, {})
        legal_items = list(self._legal_action_items())
        action_hint = ", ".join(str(item) for item in legal_items)
        action_mapping = {str(key): str(value) for key, value in ACTION_ID_MAPPING.items()}
        mapping_hint = ", ".join(f"{move_id}={label}" for move_id, label in action_mapping.items())
        view = {
            "text": f"Tick {self._tick}. Legal actions: {action_hint}\nAction mapping: {mapping_hint}",
            "vector": raw_obs,
        }
        if self._cfg.obs_image:
            frame = self._get_frame_from_env(observer_player_id=player_id)
            if frame is not None:
                view["image"] = frame
            image_history = self._get_observation_image_history(observer_player_id=player_id)
            if image_history:
                view["image_history"] = image_history
        legal_actions: Dict[str, Any] = {"items": legal_items}
        context = {
            "mode": "tick",
            "tick": self._tick,
            "step": self._tick,
            "timestamp_ms": int((time.time() - self._start_ts) * 1000),
        }
        extra = {
            "raw_obs": raw_obs,
            "info": dict(self._last_info),
            "reward": float(self._last_rewards.get(idx, 0.0)),
            "done": bool(self._done),
            "t": int(self._tick),
        }
        metadata = {
            "game_type": "vizdoom",
            "player_id": player_id,
            "player_ids": list(self._player_ids),
            "player_names": dict(self._player_names),
            "action_mapping": dict(action_mapping),
            **extra,
        }
        prompt = self._prompt_builder.build(
            game_id=self._cfg.game_id,
            active_player=player_id,
            legal_actions=legal_items,
            action_mapping=action_mapping,
            tick=self._tick,
            step=self._tick,
            last_reward=float(self._last_rewards.get(idx, 0.0)),
            view_text=str(view.get("text") or ""),
            metadata=metadata,
        )
        return ArenaObservation(
            board_text="",
            legal_moves=[str(item) for item in legal_items],
            active_player=player_id,
            metadata=metadata,
            view=view,
            legal_actions=legal_actions,
            context=context,
            last_move=None,
            prompt=prompt,
        )

    def apply(self, action: ArenaAction) -> Optional[GameResult]:
        """Apply an action and return GameResult if the game ends."""

        if not isinstance(action, ArenaAction):
            raise TypeError("apply expects ArenaAction")

        # STEP 1: Record incoming action(s).
        if isinstance(action.move, dict):
            for player_id, move in action.move.items():
                idx = self._resolve_player_index(str(player_id))
                self._pending_actions[idx] = ArenaAction(
                    player=str(player_id),
                    move=move,
                    raw=move,
                    metadata=action.metadata or {},
                )
        else:
            idx = self._resolve_player_index(str(action.player))
            self._pending_actions[idx] = action

        if not self._should_flush_actions():
            return None

        # STEP 2: Flush batched actions to the environment.
        return self._flush_actions()

    def is_terminal(self) -> bool:
        """Return True if the game has ended."""

        return self._done

    def build_result(self, *, result: str, reason: Optional[str]) -> GameResult:
        """Build a GameResult snapshot when the game ends."""

        winner = self._winner_from_info(self._last_info)
        replay_path = None
        if self._cfg.replay_in_env:
            replay_path = self._replay.finish(
                {
                    "winner": winner,
                    "result": result,
                    "reason": reason,
                }
            )
        return GameResult(
            winner=winner,
            result=result,
            reason=reason,
            move_count=self._tick,
            illegal_move_count=0,
            final_board="",
            move_log=list(self._move_log),
            replay_path=replay_path,
        )

    def set_view(self, view: str) -> None:
        """Switch the active camera view if supported by the backend."""

        normalized_view = self._normalize_pov_view(view)
        if normalized_view is not None:
            self._display_pov_view = normalized_view
        setter = getattr(self._env, "set_view", None)
        if callable(setter):
            setter(view)

    def get_last_frame(self) -> Dict[str, Any]:
        """Return latest raw frame payload for replay frame capture."""

        self._refresh_last_frame_payload()
        return dict(self._last_frame_payload)

    def close(self) -> None:
        """Close the underlying environment."""

        try:
            self._env.close()
        except Exception:
            pass

    def consume_session_progress_delta(self) -> int:
        """Report how many backend ticks completed since the last scheduler loop."""

        delta = max(0, int(self._tick) - int(self._reported_session_tick))
        self._reported_session_tick = int(self._tick)
        return delta

    def _build_env(self, cfg: ViZDoomEnvConfig):
        if cfg.use_single_process:
            backend = _load_vizdoom_backend_module(
                "gage_eval.role.arena.games.vizdoom.env_vizdoom_mp"
            )
            env_cfg = backend.ViZDoomMPEnvConfig(
                config_path=cfg.config_path,
                render_mode=cfg.render_mode,
                max_steps=cfg.max_steps,
                action_repeat=cfg.action_repeat,
                sleep_s=cfg.sleep_s,
                port=cfg.port,
            )
            return backend.ViZDoomMPEnv(env_cfg)
        backend = _load_vizdoom_backend_module(
            "gage_eval.role.arena.games.vizdoom.env_vizdoom_mp_proc"
        )
        env_cfg = backend.ViZDoomMPProcEnvConfig(
            config_path=cfg.config_path,
            render_mode=cfg.render_mode,
            pov_view=cfg.pov_view,
            show_automap=cfg.show_automap,
            automap_scale=cfg.automap_scale,
            automap_follow=cfg.automap_follow,
            automap_stride=cfg.automap_stride,
            show_pov=cfg.show_pov,
            capture_pov=cfg.capture_pov,
            pov_stride=cfg.pov_stride,
            allow_respawn=cfg.allow_respawn,
            respawn_grace_steps=cfg.respawn_grace_steps,
            no_attack_seconds=cfg.no_attack_seconds,
            max_steps=cfg.max_steps,
            action_repeat=cfg.action_repeat,
            sleep_s=cfg.sleep_s,
            port=cfg.port,
            reset_retry_count=cfg.reset_retry_count,
            death_check_warmup_steps=cfg.death_check_warmup_steps,
        )
        return backend.ViZDoomMPProcEnv(env_cfg)

    def _resolve_player_index(self, player_id: str) -> int:
        if player_id in self._player_index:
            return self._player_index[player_id]
        try:
            return int(player_id)
        except Exception:
            return 0

    def _legal_action_items(self) -> Sequence[int | str]:
        if self._cfg.action_labels:
            return [idx for idx in ALLOWED_ACTION_IDS if idx < len(self._cfg.action_labels)]
        return list(ALLOWED_ACTION_IDS)


    def _all_actions_ready(self) -> bool:
        return all(idx in self._pending_actions for idx in self._player_index.values())

    def _has_pending_actions(self) -> bool:
        return bool(self._pending_actions)

    def _should_flush_actions(self) -> bool:
        if self._all_actions_ready():
            return True
        if self._cfg.allow_partial_actions and self._has_pending_actions():
            return True
        return False

    def _flush_actions(self) -> Optional[GameResult]:
        actions: Dict[int, int] = {}
        ts_ms = int((time.time() - self._start_ts) * 1000)
        for idx in self._player_index.values():
            act = self._pending_actions.get(idx)
            if act is None:
                continue
            move_id = self._encode_move(act.move)
            actions[int(idx)] = move_id
            self._move_log.append(
                {
                    "index": self._tick,
                    "tick_index": self._tick,
                    "timestamp_ms": ts_ms,
                    "player": idx,
                    "move": move_id,
                    "raw": act.raw,
                }
            )
            if self._cfg.replay_in_env:
                self._replay.append_move(
                    {
                        "index": self._tick,
                        "tick_index": self._tick,
                        "timestamp_ms": ts_ms,
                        "actor": idx,
                        "move": move_id,
                        "raw": act.raw,
                        "event": "move",
                        "payload": {"move": move_id},
                    }
                )
        self._pending_actions = {}
        obs, rewards, done, info = self._env.step(actions)
        if self._cfg.replay_in_env:
            self._maybe_append_frame(ts_ms)
        self._obs_by_player = obs
        self._last_rewards = rewards
        self._last_info = info
        self._done = bool(done)
        self._refresh_observation_image_history()
        self._refresh_last_frame_payload()
        self._tick += 1

        if self._done:
            result = self._result_from_info(self._last_info)
            return self.build_result(result=result, reason=str(info.get("outcome") or "terminated"))
        return None

    def _maybe_append_frame(self, ts_ms: int) -> None:
        stride = int(self._cfg.frame_stride or 0)
        if stride <= 0 or self._tick % stride != 0:
            return
        frame = self._get_frame_from_env(use_display_view=True)
        if frame is None:
            return
        self._replay.append_frame(
            {
                "tick_index": self._tick,
                "timestamp_ms": ts_ms,
                "actor": self._resolve_frame_actor_id(use_display_view=True),
                "frame": frame,
            }
        )

    def _get_frame_from_env(
        self,
        *,
        observer_player_id: Optional[str] = None,
        use_display_view: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Return an encoded POV frame for the observer or configured display view."""

        _, raw = self._select_raw_pov_frame(
            observer_player_id=observer_player_id,
            use_display_view=use_display_view,
        )
        if raw is not None:
            return _encode_frame(raw)
        return None

    def _get_observation_image_history(
        self,
        *,
        observer_player_id: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        """Return encoded POV frame history for one observer in chronological order."""

        if self._cfg.obs_image_history_len <= 0:
            return []
        player_index, _ = self._select_raw_pov_frame(observer_player_id=observer_player_id)
        if player_index is None:
            return []
        history = self._obs_image_history.get(int(player_index))
        if not history:
            return []
        return [dict(frame) for frame in list(history)]

    def _refresh_observation_image_history(self) -> None:
        """Capture one encoded POV snapshot per player for future observations."""

        if not self._cfg.obs_image:
            return

        raw_frames = self._get_raw_pov_frames()
        if not raw_frames:
            return

        for player_index, raw_frame in raw_frames.items():
            encoded = _encode_frame(raw_frame)
            if encoded is None:
                continue
            history = self._obs_image_history.setdefault(
                int(player_index),
                deque(maxlen=max(1, int(self._cfg.obs_image_history_len))),
            )
            history.append(encoded)

    def _refresh_last_frame_payload(self) -> None:
        player_index, raw_frame = self._select_raw_pov_frame(use_display_view=True)
        if raw_frame is None or player_index is None:
            return
        player_id = self._player_id_by_index.get(int(player_index), f"p{player_index}")
        self._last_frame_payload = {
            "tick": int(self._tick),
            "step": int(self._tick),
            "stream_id": "pov",
            "actor": str(player_id),
            "player_index": int(player_index),
            "_rgb": raw_frame,
        }

    def _get_raw_pov_frames(self) -> Dict[int, Any]:
        frame_getter = getattr(self._env, "get_pov_frames", None)
        if callable(frame_getter):
            try:
                raw_frames = frame_getter()
            except Exception:
                raw_frames = None
            normalized = self._normalize_frame_mapping(raw_frames)
            if normalized:
                return normalized
        return self._normalize_frame_mapping(getattr(self._env, "_pov_frames", None))

    def _normalize_frame_mapping(self, raw_frames: Any) -> Dict[int, Any]:
        if not isinstance(raw_frames, dict):
            return {}
        normalized: Dict[int, Any] = {}
        for key, value in raw_frames.items():
            if value is None:
                continue
            try:
                player_index = int(key)
            except (TypeError, ValueError):
                continue
            normalized[player_index] = value
        return normalized

    def _select_raw_pov_frame(
        self,
        *,
        observer_player_id: Optional[str] = None,
        use_display_view: bool = False,
    ) -> tuple[Optional[int], Any]:
        """Select a POV frame for a specific observer or the active display view."""

        raw_frames = self._get_raw_pov_frames()
        if not raw_frames:
            return None, None

        candidate_indices: list[int] = []
        if observer_player_id:
            try:
                candidate_indices.append(self._resolve_player_index(observer_player_id))
            except KeyError:
                pass
        if use_display_view or observer_player_id is None:
            candidate_indices.extend(self._preferred_display_frame_indices())
        candidate_indices.extend(sorted(raw_frames.keys()))

        seen: set[int] = set()
        for player_index in candidate_indices:
            if player_index in seen:
                continue
            seen.add(player_index)
            raw_frame = raw_frames.get(player_index)
            if raw_frame is not None:
                return player_index, raw_frame
        return None, None

    def _preferred_display_frame_indices(self) -> list[int]:
        """Return preferred player indices for the current display POV selection."""

        view = self._display_pov_view
        if view == "p0":
            return [0]
        if view == "p1":
            return [1]
        return []

    def _resolve_frame_actor_id(self, *, use_display_view: bool = False) -> str:
        """Resolve the actor id associated with the selected frame source."""

        player_index, _ = self._select_raw_pov_frame(use_display_view=use_display_view)
        if player_index is None:
            return "p0"
        return self._player_id_by_index.get(int(player_index), f"p{player_index}")

    @staticmethod
    def _normalize_pov_view(view: Optional[str]) -> Optional[str]:
        """Normalize POV view strings accepted by the ViZDoom backend."""

        normalized = str(view or "").strip().lower()
        if normalized in {"p0", "p1", "both", "none"}:
            return normalized
        return None

    def _winner_from_info(self, info: Dict[str, Any]) -> Optional[str]:
        outcome = info.get("outcome")
        if outcome == "p0_win":
            return self._player_id_by_index.get(0, "p0")
        if outcome == "p1_win":
            return self._player_id_by_index.get(1, "p1")
        return None

    def _result_from_info(self, info: Dict[str, Any]) -> str:
        outcome = info.get("outcome")
        if outcome in ("p0_win", "p1_win"):
            return "win"
        if outcome in ("terminated", "timeout"):
            return "terminated"
        return "draw"

    def _encode_move(self, move: Any) -> int:
        if isinstance(move, int):
            return move
        if isinstance(move, str):
            stripped = move.strip()
            if stripped in self._action_label_to_id:
                return self._action_label_to_id[stripped]
            if stripped.isdigit():
                return int(stripped)
        try:
            return int(move)
        except Exception:
            return 0


class ReplayMeta:
    """Replay metadata payload."""

    def __init__(
        self,
        *,
        game_id: str,
        players: list[int],
        tick_mode: bool,
        tick_rate_hz: Optional[float],
        frame_stride: Optional[int],
        time_source: str,
    ) -> None:
        self.game_id = game_id
        self.players = players
        self.tick_mode = tick_mode
        self.tick_rate_hz = tick_rate_hz
        self.frame_stride = frame_stride
        self.time_source = time_source


class _ReplayWriter:
    """Minimal replay writer aligned with the GameArena schema."""

    def __init__(
        self,
        output_dir: Optional[str],
        *,
        run_id: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> None:
        self._output_dir = output_dir
        self._run_id = str(run_id) if run_id else None
        self._sample_id = str(sample_id) if sample_id else None
        self._meta: Optional[ReplayMeta] = None
        self._moves: list[Dict[str, Any]] = []
        self._frames: list[Dict[str, Any]] = []
        self._result: Optional[Dict[str, Any]] = None

    def start(self, meta: ReplayMeta) -> None:
        self._meta = meta
        self._moves = []
        self._frames = []
        self._result = None

    def append_move(self, move: Dict[str, Any]) -> None:
        self._moves.append(move)

    def append_frame(self, frame: Dict[str, Any]) -> None:
        self._frames.append(frame)

    def finish(self, result: Dict[str, Any]) -> Optional[str]:
        self._result = result
        path = resolve_replay_manifest_path(
            run_id=self._run_id,
            sample_id=self._sample_id,
            output_dir=self._output_dir,
        )
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self._meta.__dict__ if self._meta is not None else {},
            "moves": self._moves,
            "frames": self._frames,
            "result": self._result,
        }
        path.write_text(_json_dump(payload), encoding="utf-8")
        return str(path)




def _encode_frame(frame: Any) -> Optional[Dict[str, Any]]:
    if frame is None:
        return None
    if hasattr(frame, "tobytes") and hasattr(frame, "shape"):
        try:
            raw = frame.tobytes()
            return {
                "encoding": "raw_base64",
                "data": base64.b64encode(raw).decode("ascii"),
                "data_url": _build_image_data_url(frame),
                "shape": list(frame.shape),
                "dtype": str(getattr(frame, "dtype", "unknown")),
            }
        except Exception:
            return None
    return None


def _build_image_data_url(frame: Any) -> Optional[str]:
    if Image is None:
        return None
    try:
        image = Image.fromarray(frame)
        if image.mode not in {"RGB", "RGBA", "L"}:
            image = image.convert("RGB")
        if image.mode == "RGBA":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def _json_dump(payload: Dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def _load_vizdoom_backend_module(module_name: str):
    """Load a ViZDoom backend module with actionable error context."""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on local optional deps.
        raise ValueError(
            "ViZDoom runtime dependencies are missing. "
            "Install optional deps first (for example: pip install vizdoom pygame)."
        ) from exc
