"""Stable-retro arena environment wrapper."""

from __future__ import annotations

import base64
import io
import importlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from loguru import logger

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

from gage_eval.registry import registry
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult
from gage_eval.role.arena.games.retro.action_codec import RetroActionCodec
from gage_eval.role.arena.games.retro.observation import (
    ActionSchema,
    InfoDeltaFeeder,
    InfoFeeder,
    InfoLastFeeder,
    InfoNoneFeeder,
    ObservationBuilder,
)
from gage_eval.role.arena.games.retro.replay import ReplaySchemaWriter
from gage_eval.role.arena.games.retro.human_queue_bridge import ensure_retro_human_queue_bridge

DEFAULT_PLAYER_ID = "player_0"


@registry.asset(
    "arena_impls",
    "retro_env_v1",
    desc="Stable-retro arena environment (tick-based)",
    tags=("retro", "arena"),
)
class StableRetroArenaEnvironment:
    """Arena environment wrapper for stable-retro games."""

    def __init__(
        self,
        *,
        game: str = "SuperMarioBros3-Nes-v0",
        state: Optional[str] = None,
        default_state: str = "Start",
        rom_path: Optional[str] = None,
        player_ids: Optional[Sequence[str]] = None,
        player_names: Optional[Dict[str, str]] = None,
        runtime_policy: str = "persistent",
        display_mode: str = "headless",
        record_bk2: bool = False,
        record_dir: Optional[str] = None,
        record_filename: Optional[str] = None,
        record_path: Optional[str] = None,
        action_mapping: Optional[Dict[str, Sequence[str]]] = None,
        legal_moves: Optional[Sequence[str]] = None,
        info_feeder: Optional[Dict[str, Any]] = None,
        action_schema: Optional[Dict[str, Any]] = None,
        token_budget: int = 200,
        frame_stride: int = 1,
        snapshot_stride: int = 1,
        obs_image: bool = False,
        replay_output_dir: Optional[str] = None,
        replay_filename: Optional[str] = None,
        frame_output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        seed: Optional[int] = None,
        **_: Any,
    ) -> None:
        """Initialize the stable-retro arena environment.

        Args:
            game: Retro game id (must be imported via `retro.import`).
            state: Optional state name.
            rom_path: Optional ROM path metadata (not used by retro.make).
            player_ids: Optional list of player ids.
            player_names: Optional mapping of player ids to display names.
            runtime_policy: Runtime reuse policy (persistent or fresh).
            display_mode: Display mode (headless or pil).
            action_mapping: Optional macro-to-button mapping.
            legal_moves: Optional list of exposed macro moves.
            info_feeder: Optional info feeder config.
            action_schema: Optional action schema config.
            token_budget: Token budget for info prompt.
            frame_stride: Frame recording stride.
            snapshot_stride: Snapshot recording stride.
            obs_image: Whether to include encoded RGB image in observations.
            replay_output_dir: Optional output dir for replay schema.
            replay_filename: Optional replay filename.
            frame_output_dir: Optional output dir for frame arrays.
            run_id: Optional run identifier.
            sample_id: Optional sample identifier.
            seed: Optional reset seed.
        """

        ensure_retro_human_queue_bridge()
        self._game = str(game) if game else ""
        if not self._game:
            raise ValueError("StableRetroArenaEnvironment requires a non-empty game id")
        self._state = str(state) if state else None
        self._default_state = str(default_state or "Start")
        self._rom_path = str(rom_path) if rom_path else None
        self._runtime_policy = str(runtime_policy or "persistent").lower()
        self._seed = seed
        self._display_mode = str(display_mode or "headless").lower()
        self._render = False
        # NOTE: display_mode controls only local rendering; websocket streaming is handled
        # by an external attached server. Treat websocket modes as headless here.
        if self._display_mode in {"websocket", "ws"}:
            self._display_mode = "headless"
        self._force_frame_viewer = self._display_mode == "pil"
        self._frame_viewer_kind = "pil" if self._force_frame_viewer else "pyglet"
        self._record_bk2 = bool(record_bk2)
        self._record_dir = str(record_dir) if record_dir else None
        self._record_filename = str(record_filename) if record_filename else None
        self._record_path = str(record_path) if record_path else None
        resolved_player_ids = [str(pid) for pid in (player_ids or [DEFAULT_PLAYER_ID])]
        self._player_ids = resolved_player_ids
        self._player_names = dict(player_names or {})
        for player_id in resolved_player_ids:
            self._player_names.setdefault(player_id, player_id)
        self._active_player = resolved_player_ids[0]

        self._info_feeder = self._build_info_feeder(info_feeder)
        self._action_schema = self._build_action_schema(action_schema)
        resolved_token_budget = 200 if token_budget is None else int(token_budget)
        self._observation_builder = ObservationBuilder(
            info_feeder=self._info_feeder,
            action_schema=self._action_schema,
            token_budget=resolved_token_budget,
        )
        self._frame_stride = max(1, int(frame_stride or 1))
        self._snapshot_stride = max(1, int(snapshot_stride or 1))
        self._obs_image = bool(obs_image)
        self._replay_output_dir = replay_output_dir
        self._replay_filename = replay_filename
        self._frame_output_dir = frame_output_dir
        self._run_id = str(run_id) if run_id else None
        self._sample_id = str(sample_id) if sample_id else None

        self._retro_env = None
        self._action_codec: Optional[RetroActionCodec] = None
        self._legal_moves_override = list(legal_moves) if legal_moves else None
        self._action_mapping_override = action_mapping
        self._info_history: list[dict[str, Any]] = []
        self._reward_total = 0.0
        self._tick = 0
        self._decision_count = 0
        self._illegal_move_count = 0
        self._move_log: list[dict[str, Any]] = []
        self._last_move: Optional[str] = None
        self._last_reward = 0.0
        self._last_info: dict[str, Any] = {}
        self._last_obs = None
        self._terminal = False
        self._final_result: Optional[GameResult] = None
        self._replay_writer: Optional[ReplaySchemaWriter] = None
        self._frame_viewer = None
        self._frame_viewer_disabled = False

    def reset(self) -> None:
        self._ensure_env()
        self._reset_state()
        self._maybe_render(force=True)

    def get_active_player(self) -> str:
        return self._active_player

    def observe(self, player: str) -> ArenaObservation:
        legal_moves = self._action_codec.legal_moves() if self._action_codec else []
        controls = self._build_controls_payload(legal_moves)
        image_payload = self._build_observation_image(self._last_obs)
        return self._observation_builder.build(
            player_id=player,
            active_player=self._active_player,
            legal_moves=legal_moves,
            last_move=self._last_move,
            tick=self._tick,
            decision_count=self._decision_count,
            info_history=self._info_history,
            raw_info=self._last_info,
            reward_total=self._reward_total,
            controls=controls,
            image=image_payload,
            game_type=self._game,
            env_id=self._game,
        )

    def get_last_frame(self) -> dict[str, Any]:
        """Return the most recent structured frame payload for websocket display."""

        if self._last_obs is None:
            return {}
        # NOTE: ws_rgb live replay deduplicates snapshots by serialized frame fields,
        # so retro frames must expose tick-level metadata alongside the raw RGB array.
        metadata: dict[str, Any] = {
            "env_id": self._game,
            "tick": int(self._tick),
            "decision_count": int(self._decision_count),
            "last_reward": float(self._last_reward),
            "reward_total": float(self._reward_total),
        }
        frame_shape = self._normalize_frame_shape(getattr(self._last_obs, "shape", None))
        if frame_shape:
            metadata["frame_shape"] = frame_shape
        if self._last_info:
            metadata["info"] = dict(self._last_info)
        return {
            "env_id": self._game,
            "board_text": self._format_live_frame_text(),
            "move_count": int(self._decision_count),
            "last_move": self._last_move,
            "reward": float(self._last_reward),
            "metadata": metadata,
            "_rgb": self._last_obs,
        }

    def _build_controls_payload(self, legal_moves: Sequence[str]) -> dict[str, Any]:
        key_aliases = {
            "w": "UP",
            "a": "LEFT",
            "s": "DOWN",
            "d": "RIGHT",
            "j": "A",
            "k": "B",
            "l": "SELECT",
            "enter": "START",
        }
        button_to_key = {button: key for key, button in key_aliases.items()}
        key_order = {key: idx for idx, key in enumerate(key_aliases.keys())}

        move_aliases: dict[str, dict[str, Any]] = {}
        if self._action_codec:
            for move in legal_moves:
                try:
                    encoded = self._action_codec.encode(move)
                except Exception:
                    continue
                pressed_buttons = list(encoded.pressed)
                pressed_keys = [button_to_key.get(button) for button in pressed_buttons]
                pressed_keys = [key for key in pressed_keys if key]
                pressed_keys.sort(key=lambda key: key_order.get(key, 1_000_000))
                move_aliases[str(move)] = {
                    "buttons": pressed_buttons,
                    "keys": pressed_keys,
                    "keys_combo": "+".join(pressed_keys),
                }

        return {
            "scheme": "wasd_jkl",
            "keys_hint": "W/A/S/D = up/left/down/right; J=A(jump), K=B(run), L=select, Enter=start.",
            "key_aliases": dict(key_aliases),
            "buttons": self._action_codec.buttons() if self._action_codec else [],
            "move_aliases": move_aliases,
        }

    def _build_observation_image(self, frame: Any) -> Optional[dict[str, Any]]:
        if not self._obs_image or frame is None:
            return None
        if not hasattr(frame, "tobytes") or not hasattr(frame, "shape"):
            return None
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            return None
        normalized_shape: list[int] = []
        for dim in shape:
            try:
                normalized_shape.append(int(dim))
            except Exception:
                return None
        try:
            raw = frame.tobytes()
        except Exception:
            return None
        data_b64 = base64.b64encode(raw).decode("ascii")
        data_url = self._build_image_data_url(frame)
        return {
            "encoding": "raw_base64",
            "data": data_b64,
            "data_url": data_url,
            "shape": normalized_shape,
            "dtype": str(getattr(frame, "dtype", "unknown")),
        }

    def _format_live_frame_text(self) -> str:
        """Build a concise live frame summary for websocket display panels."""

        parts = [
            f"game={self._game}",
            f"tick={self._tick}",
            f"decisions={self._decision_count}",
            f"last_move={self._last_move or 'none'}",
            f"reward={self._last_reward:.3f}",
            f"reward_total={self._reward_total:.3f}",
        ]
        if self._last_info:
            info_text = json.dumps(self._last_info, ensure_ascii=False, sort_keys=True, default=str)
            parts.append(f"info={info_text}")
        return "\n".join(parts)

    @staticmethod
    def _normalize_frame_shape(shape: Any) -> list[int]:
        """Normalize a frame shape candidate into plain integers."""

        if shape is None:
            return []
        try:
            return [int(dim) for dim in shape]
        except Exception:
            return []

    @staticmethod
    def _build_image_data_url(frame: Any) -> Optional[str]:
        """Encode a frame into an in-memory JPEG data URL when Pillow is available."""

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

    def apply(self, action: ArenaAction) -> Optional[GameResult]:
        if self._terminal:
            return self._final_result
        if self._retro_env is None or self._action_codec is None:
            raise RuntimeError("retro environment not initialized")

        # STEP 1: Resolve hold duration and encode action once.
        hold_ticks = self._resolve_action_hold_ticks(action)
        start_tick = self._tick
        try:
            encoded = self._action_codec.encode(action.move)
        except ValueError:
            self._illegal_move_count += 1
            encoded = self._action_codec.encode("noop")

        # STEP 2: Apply repeated environment steps for the resolved hold duration.
        terminated = False
        truncated = False
        executed_ticks = 0
        while executed_ticks < hold_ticks:
            step_result = self._retro_env.step(encoded.buttons)
            obs, reward, terminated, truncated, info = self._normalize_step(step_result)
            self._last_obs = obs
            self._tick += 1
            executed_ticks += 1
            self._reward_total += float(reward)
            self._last_reward = float(reward)
            self._last_info = dict(info or {})
            self._info_history.append(self._last_info)
            if self._replay_writer is not None:
                self._replay_writer.append_tick(
                    tick=self._tick,
                    reward=float(reward),
                    info=self._last_info,
                    frame=obs,
                    done=bool(terminated or truncated),
                )
            self._maybe_render()
            if terminated or truncated:
                break

        self.record_decision(action, start_tick=start_tick, hold_ticks=max(1, executed_ticks))

        # STEP 3: Check terminal state.
        if not (terminated or truncated):
            return None
        self._terminal = True
        status = self._derive_result(terminated, truncated, self._last_info)
        reason = "truncated" if truncated else "terminated"
        self._final_result = self.build_result(status=status, reason=reason)
        if self._replay_writer is not None:
            replay_path = self._replay_writer.finalize(self._final_result)
            if replay_path:
                self._final_result = GameResult(
                    winner=self._final_result.winner,
                    result=self._final_result.result,
                    reason=self._final_result.reason,
                    replay_path=replay_path,
                    move_count=self._final_result.move_count,
                    illegal_move_count=self._final_result.illegal_move_count,
                    final_board=self._final_result.final_board,
                    move_log=self._final_result.move_log,
                    rule_profile=self._final_result.rule_profile,
                    win_direction=self._final_result.win_direction,
                    line_length=self._final_result.line_length,
                )
        return self._final_result

    def is_terminal(self) -> bool:
        return self._terminal

    def build_result(
        self,
        *,
        status: Optional[str] = None,
        result: Optional[str] = None,
        reason: Optional[str],
    ) -> GameResult:
        final_board = self._format_final_board()
        resolved_status = str(status or result or "draw")
        return GameResult(
            winner=self._active_player if resolved_status == "win" else None,
            result=resolved_status,
            reason=reason,
            replay_path=None,
            move_count=self._decision_count,
            illegal_move_count=self._illegal_move_count,
            final_board=final_board,
            move_log=list(self._move_log),
        )

    def finalize_replay(self, result: GameResult) -> GameResult:
        """Finalize replay artifacts for a result.

        Args:
            result: Result dataclass to attach replay_path to.

        Returns:
            Updated GameResult with replay_path when available.
        """

        if self._replay_writer is None:
            return result
        replay_path = self._replay_writer.finalize(result)
        if not replay_path:
            return result
        return GameResult(
            winner=result.winner,
            result=result.result,
            reason=result.reason,
            replay_path=replay_path,
            move_count=result.move_count,
            illegal_move_count=result.illegal_move_count,
            final_board=result.final_board,
            move_log=result.move_log,
            rule_profile=result.rule_profile,
            win_direction=result.win_direction,
            line_length=result.line_length,
        )

    def record_decision(
        self,
        action: ArenaAction,
        *,
        start_tick: int,
        hold_ticks: int,
        latency_ms: Optional[int] = None,
        timed_out: bool = False,
        fallback_used: Optional[str] = None,
        llm_wait_mode: Optional[str] = None,
    ) -> None:
        end_tick = max(start_tick, start_tick + hold_ticks - 1)
        decision_index = self._decision_count
        metadata = dict(action.metadata or {})
        entry = {
            "decision_index": decision_index,
            "move": action.move,
            "raw": action.raw,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "hold_ticks": hold_ticks,
            "latency_ms": latency_ms if latency_ms is not None else metadata.get("latency_ms"),
            "timed_out": bool(timed_out or metadata.get("timed_out", False)),
            "fallback_used": fallback_used if fallback_used is not None else metadata.get("fallback_used"),
            "llm_wait_mode": llm_wait_mode if llm_wait_mode is not None else metadata.get("llm_wait_mode"),
        }
        self._move_log.append(entry)
        self._decision_count += 1
        self._last_move = action.move
        if self._replay_writer is not None:
            self._replay_writer.append_decision(
                action,
                decision_index=decision_index,
                start_tick=start_tick,
                end_tick=end_tick,
            )

    def _resolve_action_hold_ticks(self, action: ArenaAction) -> int:
        metadata = action.metadata if isinstance(action.metadata, dict) else {}
        candidate = metadata.get("hold_ticks")
        if candidate is None:
            candidate = self._extract_hold_ticks_from_raw(action.raw)
        if candidate is None:
            hold_ticks = 1
        else:
            try:
                hold_ticks = int(candidate)
            except (TypeError, ValueError):
                hold_ticks = 1
        return max(self._action_schema.hold_ticks_min, min(self._action_schema.hold_ticks_max, hold_ticks))

    @staticmethod
    def _extract_hold_ticks_from_raw(raw: Any) -> Optional[int]:
        if not isinstance(raw, str):
            return None
        stripped = raw.strip()
        if not stripped:
            return None
        if not stripped.startswith("{"):
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        value = payload.get("hold_ticks")
        if value is None:
            value = payload.get("holdTicks")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _ensure_env(self) -> None:
        if self._retro_env is None or self._runtime_policy == "fresh":
            self._retro_env = self._make_env()
            self._action_codec = self._build_action_codec(self._retro_env)

    def _reset_state(self) -> None:
        if self._retro_env is None:
            raise RuntimeError("retro environment not initialized")
        self._info_history = []
        self._reward_total = 0.0
        self._tick = 0
        self._decision_count = 0
        self._illegal_move_count = 0
        self._move_log = []
        self._last_move = None
        self._last_reward = 0.0
        self._last_info = {}
        self._terminal = False
        self._final_result = None
        self._replay_writer = ReplaySchemaWriter(
            game=self._game,
            state=self._state,
            run_id=self._run_id,
            sample_id=self._sample_id,
            replay_output_dir=self._replay_output_dir,
            replay_filename=self._replay_filename,
            frame_output_dir=self._frame_output_dir,
            frame_stride=self._frame_stride,
            snapshot_stride=self._snapshot_stride,
            rom_path=self._rom_path,
        )

        if self._seed is not None:
            try:
                reset_result = self._retro_env.reset(seed=self._seed)
            except TypeError:
                reset_result = self._retro_env.reset()
        else:
            reset_result = self._retro_env.reset()
        obs, info = self._normalize_reset(reset_result)
        self._last_obs = obs
        if info:
            self._last_info = dict(info)
            self._info_history.append(self._last_info)
        if self._replay_writer is not None:
            self._replay_writer.append_tick(
                tick=self._tick,
                reward=0.0,
                info=self._last_info,
                frame=obs,
                done=False,
            )

    def _make_env(self):
        # STEP 1: Set up render mode hints before importing retro.
        os.environ.setdefault("PYGLET_HEADLESS", "true")
        logger.info(
            "Retro render setup: display_mode={} display={} headless={}",
            self._display_mode,
            os.environ.get("DISPLAY"),
            os.environ.get("PYGLET_HEADLESS"),
        )
        try:
            retro = importlib.import_module("retro")
        except ImportError as exc:
            raise ImportError(
                "stable-retro is required. Install it and import the ROM via `python -m retro.import`."
            ) from exc
        record_target = self._resolve_record_output_path()
        if record_target is not None:
            try:
                record_target.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                record_target = None
        state = self._state or self._default_state
        render_mode = "rgb_array"
        logger.info("Retro render mode: {}", render_mode)
        # STEP 2: Build the retro environment.
        try:
            if record_target is not None:
                return retro.make(game=self._game, state=state, record=str(record_target), render_mode=render_mode)
            return retro.make(game=self._game, state=state, render_mode=render_mode)
        except TypeError:
            try:
                if record_target is not None:
                    return retro.make(game=self._game, record=str(record_target), render_mode=render_mode)
                return retro.make(game=self._game, render_mode=render_mode)
            except TypeError:
                if record_target is not None:
                    logger.warning("retro.make does not support bk2 recording for this build")
                return retro.make(game=self._game)

    def _maybe_render(self, *, force: bool = False) -> None:
        if not self._force_frame_viewer:
            return
        if self._retro_env is None:
            return
        try:
            logger.debug("Retro render tick={} force={}", self._tick, force)
            if self._last_obs is None:
                if self._tick <= 3:
                    logger.warning("Force frame viewer enabled but no frame is available yet.")
            else:
                self._show_frame(self._last_obs)
        except Exception:
            return

    def _resolve_record_output_path(self) -> Optional[Path]:
        if self._record_path:
            return Path(self._record_path)
        if not self._record_bk2:
            return None
        base_dir = None
        if self._record_dir:
            base_dir = Path(self._record_dir)
        elif self._run_id:
            base_dir = Path(os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs")) / self._run_id / "replays"
        if base_dir is None:
            return None
        sample_id_source = self._sample_id or os.environ.get("GAGE_EVAL_SAMPLE_ID") or "unknown"
        sample_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(sample_id_source)).strip("_") or "unknown"
        filename = self._record_filename or f"retro_movie_{sample_id}.bk2"
        return base_dir / filename

    def _build_action_codec(self, retro_env) -> RetroActionCodec:
        buttons = list(getattr(retro_env, "buttons", []))
        if not buttons:
            buttons = list(getattr(retro_env.unwrapped, "buttons", [])) if hasattr(retro_env, "unwrapped") else []
        if not buttons:
            raise ValueError("retro env missing buttons list")
        return RetroActionCodec(
            buttons=buttons,
            macro_map=self._action_mapping_override,
            legal_moves=self._legal_moves_override,
        )

    def _show_frame(self, frame) -> None:
        if frame is None:
            return
        frame_shape = getattr(frame, "shape", None)
        frame_ndim = getattr(frame, "ndim", 0)
        if frame_ndim != 3:
            logger.warning("Frame viewer skipped non-RGB frame: ndim={} shape={}", frame_ndim, frame_shape)
            return
        if self._tick <= 3 or self._tick % 60 == 0:
            try:
                min_val = float(frame.min())
                max_val = float(frame.max())
            except Exception:
                min_val = None
                max_val = None
            logger.debug(
                "Frame viewer stats: shape={} dtype={} min={} max={}",
                frame_shape,
                getattr(frame, "dtype", None),
                min_val,
                max_val,
            )
        if self._frame_viewer_disabled:
            return
        if self._frame_viewer is None:
            self._frame_viewer = self._init_frame_viewer()
            if self._frame_viewer is None:
                self._frame_viewer_disabled = True
                return
        try:
            if self._frame_viewer_kind == "pil":
                self._frame_viewer["image"] = self._frame_viewer["photo_factory"](frame)
                self._frame_viewer["label"].configure(image=self._frame_viewer["image"])
                self._frame_viewer["root"].update_idletasks()
                self._frame_viewer["root"].update()
            else:
                self._frame_viewer.imshow(frame)
        except Exception:
            return

    def _init_frame_viewer(self):
        if self._frame_viewer_kind == "pil":
            try:
                import tkinter as tk
                from PIL import Image, ImageTk
            except Exception:
                logger.warning("PIL frame viewer unavailable; install pillow and tkinter to enable it.")
                return None
            root = tk.Tk()
            root.title("Retro Frame Viewer")
            label = tk.Label(root)
            label.pack()

            def photo_factory(arr):
                image = Image.fromarray(arr)
                return ImageTk.PhotoImage(image)

            return {"root": root, "label": label, "photo_factory": photo_factory, "image": None}
        try:
            from stable_retro.rendering import SimpleImageViewer
        except Exception:
            logger.warning("Pyglet frame viewer unavailable.")
            return None
        return SimpleImageViewer()

    def _build_info_feeder(self, info_feeder: Optional[Dict[str, Any]]) -> InfoFeeder:
        if not info_feeder:
            return InfoLastFeeder()
        impl = str(info_feeder.get("impl") or info_feeder.get("name") or "info_last_v1").lower()
        params = dict(info_feeder.get("params") or {})
        if impl in {"info_none_v1", "none", "off", "disabled"}:
            return InfoNoneFeeder()
        if impl in {"info_delta_v1", "delta", "info_delta"}:
            return InfoDeltaFeeder(window_size=int(params.get("window_size", 8) or 8))
        return InfoLastFeeder()

    def _build_action_schema(self, action_schema: Optional[Dict[str, Any]]) -> ActionSchema:
        if not action_schema:
            return ActionSchema()
        return ActionSchema(
            hold_ticks_min=int(action_schema.get("hold_ticks_min", 1) or 1),
            hold_ticks_max=int(action_schema.get("hold_ticks_max", 20) or 20),
            default_hold_ticks=int(action_schema.get("hold_ticks_default", 6) or 6),
        )

    @staticmethod
    def _normalize_step(step_result):
        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            return obs, reward, terminated, truncated, info
        if isinstance(step_result, tuple) and len(step_result) == 4:
            obs, reward, done, info = step_result
            return obs, reward, done, False, info
        raise ValueError("retro step returned unexpected format")

    @staticmethod
    def _normalize_reset(reset_result):
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
            return obs, info
        return reset_result, {}

    @staticmethod
    def _derive_result(terminated: bool, truncated: bool, info: Dict[str, Any]) -> str:
        if truncated:
            return "draw"
        for key in ("win", "won", "victory", "level_complete", "is_level_complete"):
            value = info.get(key)
            if isinstance(value, bool) and value:
                return "win"
        return "loss" if terminated else "draw"

    def _format_final_board(self) -> str:
        summary = {
            "tick": self._tick,
            "reward_total": self._reward_total,
            "last_reward": self._last_reward,
            "last_info": self._last_info,
        }
        return str(summary)
