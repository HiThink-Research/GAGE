from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import multiprocessing as mp
import os
import random
import time

from loguru import logger

try:
    import vizdoom  # type: ignore
    from vizdoom import AutomapMode, DoomGame, GameVariable, Mode  # type: ignore
    _VIZDOOM_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on local optional deps.
    vizdoom = None  # type: ignore[assignment]
    AutomapMode = DoomGame = GameVariable = Mode = Any  # type: ignore[assignment]
    _VIZDOOM_IMPORT_ERROR = exc

_AUTOMAP_VIZ_MODULE: Any = None
_MAX_STALE_RESET_RECOVERY_RESTARTS = 2


@dataclass
class ViZDoomMPProcEnvConfig:
    config_path: Optional[str] = None
    render_mode: Optional[str] = None  # None | "p0" | "p1" | "both"
    pov_view: Optional[str] = None  # None | "p0" | "p1" | "both" | "none"
    show_automap: bool = False
    automap_scale: int = 10
    automap_follow: bool = True
    automap_stride: int = 2
    show_pov: bool = True
    pov_stride: int = 2
    allow_respawn: bool = True
    respawn_grace_steps: int = 60
    no_attack_seconds: float = 10.0
    max_steps: int = 500
    action_repeat: int = 1
    sleep_s: float = 0.0
    port: Optional[int] = None
    reset_retry_count: int = 3
    death_check_warmup_steps: int = 8


def _resolve_cfg_path(config_path: Optional[str]) -> str:
    _ensure_vizdoom_available()
    if config_path:
        if os.path.isabs(config_path):
            return config_path
        return os.path.abspath(config_path)
    pkg_dir = os.path.dirname(vizdoom.__file__)
    return os.path.join(pkg_dir, "scenarios", "multi_duel.cfg")
    # return os.path.join(pkg_dir, "scenarios", "timidity.cfg")
    # return os.path.join(pkg_dir, "scenarios", "basic.cfg")


def _build_action_map(game: DoomGame) -> Dict[int, List[int]]:
    n = game.get_available_buttons_size()
    btns = [b.name for b in game.get_available_buttons()]
    idx = {name: i for i, name in enumerate(btns)}

    def vec(*ones_idx: int) -> List[int]:
        a = [0] * n
        for i in ones_idx:
            if 0 <= i < n:
                a[i] = 1
        return a

    def safe(name: str) -> Optional[int]:
        return idx.get(name)

    noop = vec()
    attack = vec(*( [safe("ATTACK")] if safe("ATTACK") is not None else [] ))
    turn_l = vec(*( [safe("TURN_LEFT")] if safe("TURN_LEFT") is not None else [] ))
    turn_r = vec(*( [safe("TURN_RIGHT")] if safe("TURN_RIGHT") is not None else [] ))
    forward = vec(*( [safe("MOVE_FORWARD")] if safe("MOVE_FORWARD") is not None else [] ))

    if sum(turn_l) == 0 and safe("MOVE_LEFT") is not None:
        turn_l = vec(safe("MOVE_LEFT"))
    if sum(turn_r) == 0 and safe("MOVE_RIGHT") is not None:
        turn_r = vec(safe("MOVE_RIGHT"))

    return {
        0: noop,
        1: attack,
        2: turn_l,
        3: turn_r,
        4: forward,
    }


def _get_obs(game: DoomGame, game_vars: List[GameVariable], t: int) -> Dict[str, Any]:
    obs: Dict[str, Any] = {"t": t}
    try:
        obs["total_reward"] = float(game.get_total_reward())
    except Exception:
        pass

    for var in game_vars:
        try:
            obs[var.name] = float(game.get_game_variable(var))
        except Exception:
            pass
    return obs


def _downsample(frame, factor: int = 2):
    if frame is None or factor <= 1:
        return frame
    try:
        return frame[::factor, ::factor, :]
    except Exception:
        return frame


def _worker(
    conn: mp.connection.Connection,
    role: str,
    cfg_path: str,
    port: int,
    render_mode: Optional[str],
    show_automap: bool,
    automap_scale: int,
    automap_follow: bool,
    automap_stride: int,
    show_pov: bool,
    pov_stride: int,
    allow_respawn: bool,
    respawn_grace_steps: int,
    no_attack_seconds: float,
    action_repeat: int,
) -> None:
    _ensure_vizdoom_available()
    game: Optional[DoomGame] = None
    try:
        game = DoomGame()
        game.load_config(cfg_path)

        if role == "host":
            host_args = f"-host 2 -port {port} -deathmatch +timelimit 1 +sv_spawnfarthest 1 +viz_connect_timeout 120"
            game.add_game_args(host_args)
        else:
            time.sleep(0.5)
            join_args = f"-join 127.0.0.1 -port {port}"
            game.add_game_args(join_args)

        # Align with official examples: register common game variables and quality settings.
        game.add_available_game_variable(GameVariable.FRAGCOUNT)
        if hasattr(GameVariable, "SCORE"):
            game.add_available_game_variable(GameVariable.SCORE)
        game.add_available_game_variable(GameVariable.PLAYER_NUMBER)
        if hasattr(GameVariable, "HEALTH"):
            game.add_available_game_variable(GameVariable.HEALTH)
        if hasattr(GameVariable, "POSITION_X"):
            game.add_available_game_variable(GameVariable.POSITION_X)
        if hasattr(GameVariable, "POSITION_Y"):
            game.add_available_game_variable(GameVariable.POSITION_Y)
        if hasattr(GameVariable, "POSITION_Z"):
            game.add_available_game_variable(GameVariable.POSITION_Z)
        if hasattr(GameVariable, "ANGLE"):
            game.add_available_game_variable(GameVariable.ANGLE)

        if show_automap:
            game.set_automap_buffer_enabled(True)
            game.set_automap_mode(AutomapMode.OBJECTS_WITH_SIZE)
            game.add_game_args(
                f"+am_followplayer {1 if automap_follow else 0} +viz_am_scale {int(automap_scale)} "
                "+am_showplayers 1 +am_showallplayers 1 +am_cheat 0 +am_textured 0"
            )

        # Match basic.py visual quality.
        # game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        game.set_screen_resolution(vizdoom.ScreenResolution.RES_800X600)
        game.set_screen_format(vizdoom.ScreenFormat.RGB24)
        game.set_render_hud(True)
        game.set_render_minimal_hud(True)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.set_render_effects_sprites(False)
        game.set_render_messages(False)
        game.set_render_corpses(False)
        game.set_render_screen_flashes(True)

        game.set_mode(Mode.ASYNC_PLAYER)
        if render_mode is None:
            visible = False
        elif render_mode == "both":
            visible = True
        elif render_mode == "p0":
            visible = role == "host"
        elif render_mode == "p1":
            visible = role == "join"
        else:
            visible = False
        game.set_window_visible(visible)

        game.init()
        action_map = _build_action_map(game)
        attack_idx = None
        for i, b in enumerate(game.get_available_buttons()):
            if b.name == "ATTACK":
                attack_idx = i
                break
        no_attack_steps = max(0, int(no_attack_seconds * 35 / max(1, action_repeat)))
        game_vars = list(game.get_available_game_variables())
        conn.send({"type": "ready", "game_vars": [v.name for v in game_vars]})

        while True:
            if not conn.poll(0.1):
                continue
            msg = conn.recv()
            cmd = msg.get("type")
            if cmd == "reset":
                seed = msg.get("seed")
                if seed is not None:
                    try:
                        game.set_seed(int(seed))
                    except Exception:
                        pass
                game.new_episode()
                try:
                    game.advance_action(1)
                except Exception:
                    pass
                if show_automap:
                    state = game.get_state()
                    print(
                        "[DEBUG] reset automap:",
                        None if state is None or state.automap_buffer is None else state.automap_buffer.shape,
                        flush=True,
                    )
                t = int(msg.get("t", 0))
                obs = _get_obs(game, game_vars, t)
                payload = {"type": "reset", "obs": obs, "done": game.is_episode_finished()}
                if show_automap:
                    state = game.get_state()
                    payload["automap"] = None if state is None else state.automap_buffer
                if show_pov:
                    state = game.get_state()
                    if state is not None and state.screen_buffer is not None:
                        payload["screen"] = state.screen_buffer
                conn.send(payload)
            elif cmd == "step":
                t = int(msg.get("t", 0))
                if game.is_episode_finished():
                    if allow_respawn and (respawn_grace_steps <= 0 or t < respawn_grace_steps):
                        try:
                            game.respawn_player()
                        except Exception:
                            pass
                    if not game.is_episode_finished():
                        obs = _get_obs(game, game_vars, t)
                        payload = {"type": "step", "obs": obs, "reward": 0.0, "done": False}
                        if show_automap:
                            state = game.get_state()
                            if state is not None and state.automap_buffer is not None:
                                payload["automap"] = state.automap_buffer
                        conn.send(payload)
                        continue
                    obs = _get_obs(game, game_vars, t)
                    payload = {"type": "step", "obs": obs, "reward": 0.0, "done": True}
                    if show_automap:
                        state = game.get_state()
                        if state is not None and state.automap_buffer is not None:
                            payload["automap"] = state.automap_buffer
                    if show_pov:
                        state = game.get_state()
                        if state is not None and state.screen_buffer is not None:
                            payload["screen"] = state.screen_buffer
                    conn.send(payload)
                    continue
                action = int(msg.get("action", 0))
                vec = action_map.get(action, action_map[0])
                if attack_idx is not None and t < no_attack_steps:
                    vec = list(vec)
                    vec[attack_idx] = 0
                reward = float(game.make_action(vec, action_repeat))
                done = game.is_episode_finished()
                if done and allow_respawn and (respawn_grace_steps <= 0 or t < respawn_grace_steps):
                    try:
                        game.respawn_player()
                        done = game.is_episode_finished()
                    except Exception:
                        pass
                obs = _get_obs(game, game_vars, t)
                payload = {"type": "step", "obs": obs, "reward": reward, "done": done}
                if show_automap and (automap_stride <= 1 or (t % automap_stride == 0)):
                    state = game.get_state()
                    payload["automap"] = None if state is None else state.automap_buffer
                if show_pov and (pov_stride <= 1 or (t % pov_stride == 0)):
                    state = game.get_state()
                    if state is not None and state.screen_buffer is not None:
                        payload["screen"] = state.screen_buffer
                conn.send(payload)
            elif cmd == "set_visible":
                visible = bool(msg.get("visible", False))
                game.set_window_visible(visible)
            elif cmd == "close":
                break
    except Exception as e:
        try:
            conn.send({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            if game is not None:
                game.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def _ensure_vizdoom_available() -> None:
    if vizdoom is not None:
        return
    raise RuntimeError(
        "ViZDoom backend is unavailable because optional dependency 'vizdoom' is not installed."
    ) from _VIZDOOM_IMPORT_ERROR


def _load_automap_viz_module() -> Any:
    """Load automap visualization helpers lazily to avoid unnecessary SDL conflicts."""

    global _AUTOMAP_VIZ_MODULE
    if _AUTOMAP_VIZ_MODULE is None:
        from . import automap_viz as automap_viz_module

        _AUTOMAP_VIZ_MODULE = automap_viz_module
    return _AUTOMAP_VIZ_MODULE


class ViZDoomMPProcEnv:
    """
    Multiprocess multiplayer ViZDoom env adapter (2 players, host/join).
    - reset(seed) -> {pid: obs}
    - step({pid: action_int}) -> (obs_by_pid, rewards_by_pid, done, info)
    """
    def __init__(self, config: ViZDoomMPProcEnvConfig):
        self.cfg = config
        self.t = 0
        self.port: Optional[int] = None
        self.players = (0, 1)
        self._automap_p0 = None
        self._automap_p1 = None
        self._pov = None
        self._pov_frames = {0: None, 1: None}
        self._view = self.cfg.render_mode
        self._round_done = False
        self.match_count = 0
        self.scores = {0: 0, 1: 0}
        self._last_health: Dict[int, Optional[float]] = {0: None, 1: None}
        self._seen_alive: Dict[int, bool] = {0: False, 1: False}

        if self.cfg.render_mode is None and self.cfg.show_automap:
            self.cfg.show_automap = False

        self._ctx = mp.get_context("spawn")
        self._host_proc: Optional[mp.Process] = None
        self._join_proc: Optional[mp.Process] = None
        self._host_conn: Optional[mp.connection.Connection] = None
        self._join_conn: Optional[mp.connection.Connection] = None

    def _select_port(self, attempt: int) -> int:
        if attempt == 0 and self.cfg.port is not None:
            return int(self.cfg.port)
        return random.randint(5200, 5900)

    def _recv_with_timeout(self, conn: mp.connection.Connection, timeout_s: float) -> Dict[str, Any]:
        if not conn.poll(timeout_s):
            raise TimeoutError("Timeout waiting for child process message.")
        return conn.recv()

    def _start_processes(self) -> None:
        cfg_path = _resolve_cfg_path(self.cfg.config_path)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"ViZDoom cfg not found: {cfg_path}")

        last_err: Optional[Exception] = None
        for attempt in range(5):
            port = self._select_port(attempt)
            host_args = f"-host 2 -port {port} -deathmatch +timelimit 1 +sv_spawnfarthest 1 +viz_connect_timeout 120"
            join_args = f"-join 127.0.0.1 -port {port}"
            print(f"[ViZDoomMPProc] cfg={cfg_path} port={port}")
            print(f"[ViZDoomMPProc] host_args='{host_args}'")
            print(f"[ViZDoomMPProc] join_args='{join_args}'")

            host_parent, host_child = self._ctx.Pipe(duplex=True)
            join_parent, join_child = self._ctx.Pipe(duplex=True)

            host_proc = self._ctx.Process(
                target=_worker,
                args=(
                    host_child,
                    "host",
                    cfg_path,
                    port,
                    self.cfg.render_mode,
                    self.cfg.show_automap,
                    self.cfg.automap_scale,
                    self.cfg.automap_follow,
                    self.cfg.automap_stride,
                    self.cfg.show_pov,
                    self.cfg.pov_stride,
                    self.cfg.allow_respawn,
                    self.cfg.respawn_grace_steps,
                    self.cfg.no_attack_seconds,
                    self.cfg.action_repeat,
                ),
                daemon=True,
            )
            join_proc = self._ctx.Process(
                target=_worker,
                args=(
                    join_child,
                    "join",
                    cfg_path,
                    port,
                    self.cfg.render_mode,
                    self.cfg.show_automap,
                    self.cfg.automap_scale,
                    self.cfg.automap_follow,
                    self.cfg.automap_stride,
                    self.cfg.show_pov,
                    self.cfg.pov_stride,
                    self.cfg.allow_respawn,
                    self.cfg.respawn_grace_steps,
                    self.cfg.no_attack_seconds,
                    self.cfg.action_repeat,
                ),
                daemon=True,
            )

            host_proc.start()
            join_proc.start()

            try:
                host_msg = self._recv_with_timeout(host_parent, 140.0)
                if host_msg.get("type") != "ready":
                    raise RuntimeError(f"Host init failed: {host_msg}")
                print("[ViZDoomMPProc] available_game_vars:", host_msg.get("game_vars", []), flush=True)
                join_msg = self._recv_with_timeout(join_parent, 140.0)
                if join_msg.get("type") != "ready":
                    raise RuntimeError(f"Join init failed: {join_msg}")
            except Exception as e:
                last_err = e
                self._terminate_process(join_proc, join_parent)
                self._terminate_process(host_proc, host_parent)
                time.sleep(0.2)
                continue

            self._host_proc = host_proc
            self._join_proc = join_proc
            self._host_conn = host_parent
            self._join_conn = join_parent
            self.port = port
            return

        raise RuntimeError(f"Failed to init multiplayer ViZDoom after retries: {last_err}")

    def _terminate_process(self, proc: Optional[mp.Process], conn: Optional[mp.connection.Connection]) -> None:
        if proc is None and conn is None:
            return
        proc_pid = getattr(proc, "pid", None)
        proc_alive = bool(proc is not None and proc.is_alive())
        logger.info(
            "ViZDoomMPProc terminate start pid={} alive={} conn_open={}",
            proc_pid,
            proc_alive,
            conn is not None,
        )
        try:
            if conn is not None:
                try:
                    conn.send({"type": "close"})
                except Exception as exc:
                    logger.warning("ViZDoomMPProc conn send close failed pid={} err={}", proc_pid, exc)
                try:
                    conn.close()
                except Exception as exc:
                    logger.warning("ViZDoomMPProc conn close failed pid={} err={}", proc_pid, exc)
        finally:
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        proc_alive_after = bool(proc is not None and proc.is_alive())
        logger.info("ViZDoomMPProc terminate done pid={} alive={}", proc_pid, proc_alive_after)

    def close(self) -> None:
        logger.info(
            "ViZDoomMPProc close start host_pid={} join_pid={}",
            getattr(self._host_proc, "pid", None),
            getattr(self._join_proc, "pid", None),
        )
        self._terminate_process(self._join_proc, self._join_conn)
        self._terminate_process(self._host_proc, self._host_conn)
        self._join_proc = None
        self._host_proc = None
        self._join_conn = None
        self._host_conn = None
        if self._automap_p0 is not None:
            self._automap_p0.close()
            self._automap_p0 = None
        if self._automap_p1 is not None:
            self._automap_p1.close()
            self._automap_p1 = None
        if self._pov is not None:
            self._pov.close()
            self._pov = None
        logger.info("ViZDoomMPProc close done")

    def _dispose_runtime_processes(self) -> None:
        """Terminate host/join child processes and clear connection state."""

        self._terminate_process(self._join_proc, self._join_conn)
        self._terminate_process(self._host_proc, self._host_conn)
        self._join_proc = None
        self._host_proc = None
        self._join_conn = None
        self._host_conn = None
        self.port = None

    def _reset_with_existing_processes(
        self,
        *,
        seed: Optional[int],
        attempts: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Reset current host/join processes and return whether stale state persists."""

        assert self._host_conn is not None
        assert self._join_conn is not None

        host_msg: Dict[str, Any] = {}
        join_msg: Dict[str, Any] = {}
        stale_reset = True
        for attempt in range(1, attempts + 1):
            self._host_conn.send({"type": "reset", "seed": seed, "t": self.t})
            self._join_conn.send({"type": "reset", "seed": seed, "t": self.t})

            host_msg = self._recv_with_timeout(self._host_conn, 20.0)
            print(
                "[DEBUG] reset host automap:",
                None if host_msg.get("automap") is None else host_msg["automap"].shape,
                flush=True,
            )
            join_msg = self._recv_with_timeout(self._join_conn, 20.0)
            if host_msg.get("type") == "error":
                raise RuntimeError(f"Host reset failed: {host_msg.get('error')}")
            if join_msg.get("type") == "error":
                raise RuntimeError(f"Join reset failed: {join_msg.get('error')}")

            host_obs = host_msg.get("obs", {})
            join_obs = join_msg.get("obs", {})
            host_done = bool(host_msg.get("done", False))
            join_done = bool(join_msg.get("done", False))
            self._update_last_health(0, host_obs)
            self._update_last_health(1, join_obs)

            health0 = self._effective_health(0, host_obs)
            health1 = self._effective_health(1, join_obs)
            stale_reset = host_done or join_done or self._is_dead(health0) or self._is_dead(health1)
            logger.info(
                "ViZDoomMPProc reset attempt={} host_done={} join_done={} p0_health={} p1_health={} stale={}",
                attempt,
                host_done,
                join_done,
                health0,
                health1,
                stale_reset,
            )
            if not stale_reset:
                break
            if attempt < attempts:
                time.sleep(0.15)
        return host_msg, join_msg, stale_reset

    def set_view(self, view: str) -> None:
        if view not in ("p0", "p1", "both", "none"):
            return
        self._view = view
        if self._host_conn is None or self._join_conn is None:
            return
        if view == "p0":
            host_vis, join_vis = True, False
        elif view == "p1":
            host_vis, join_vis = False, True
        elif view == "both":
            host_vis, join_vis = True, True
        else:
            host_vis, join_vis = False, False
        try:
            self._host_conn.send({"type": "set_visible", "visible": host_vis})
            self._join_conn.send({"type": "set_visible", "visible": join_vis})
        except Exception:
            pass

    def _toggle_view(self) -> None:
        if self._view in (None, "p0"):
            self.set_view("p1")
        elif self._view == "p1":
            self.set_view("p0")
        elif self._view == "both":
            self.set_view("p0")
        else:
            self.set_view("p0")

    def reset(self, seed: Optional[int] = None) -> Dict[int, Any]:
        if self._host_proc is None or self._join_proc is None:
            self._start_processes()
        assert self._host_conn is not None
        assert self._join_conn is not None

        # STEP 1: Reset round state and initialize optional debug windows.
        self.t = 0
        self._round_done = False
        self._last_health = {0: None, 1: None}
        self._seen_alive = {0: False, 1: False}
        automap_viz_module = None
        if self.cfg.show_automap and self._automap_p0 is None:
            automap_viz_module = automap_viz_module or _load_automap_viz_module()
            self._automap_p0 = automap_viz_module.init_window(window_name="Automap (p0)", bgr=False)
        if self.cfg.show_automap and self._automap_p1 is None:
            automap_viz_module = automap_viz_module or _load_automap_viz_module()
            self._automap_p1 = automap_viz_module.init_window(window_name="Automap (p1)", bgr=False)
        if self.cfg.show_pov and self._pov is None:
            automap_viz_module = automap_viz_module or _load_automap_viz_module()
            self._pov = automap_viz_module.init_window(window_name="ViZDoom POV", bgr=False)

        # STEP 2: Retry reset when stale intermission/death state leaks into the new round.
        host_msg: Dict[str, Any] = {}
        join_msg: Dict[str, Any] = {}
        attempts = max(1, int(self.cfg.reset_retry_count))
        stale_reset = True
        for restart_idx in range(_MAX_STALE_RESET_RECOVERY_RESTARTS + 1):
            if self._host_proc is None or self._join_proc is None:
                self._start_processes()
            host_msg, join_msg, stale_reset = self._reset_with_existing_processes(
                seed=seed,
                attempts=attempts,
            )
            if not stale_reset:
                break
            logger.warning(
                "ViZDoomMPProc reset remained stale after {} attempts; restarting workers ({}/{})",
                attempts,
                restart_idx + 1,
                _MAX_STALE_RESET_RECOVERY_RESTARTS + 1,
            )
            if restart_idx >= _MAX_STALE_RESET_RECOVERY_RESTARTS:
                self._dispose_runtime_processes()
                raise RuntimeError(
                    "ViZDoom reset failed: stale death/intermission state persisted after process restarts."
                )
            self._dispose_runtime_processes()
            time.sleep(0.2)

        logger.info(
            "ViZDoomMPProc reset done host_done={} join_done={} host_obs={} join_obs={}",
            host_msg.get("done"),
            join_msg.get("done"),
            host_msg.get("obs", {}),
            join_msg.get("obs", {}),
        )

        if self.cfg.show_automap:
            frame0 = host_msg.get("automap")
            frame1 = join_msg.get("automap")
            if self._automap_p0 is not None and frame0 is not None:
                key = self._automap_p0.update(frame0)
                if key in (13, ord("v"), ord("V")):
                    self._toggle_view()
                    print("[ViZDoomMPProc] toggled view:", self._view, flush=True)
            if self._automap_p1 is not None and frame1 is not None:
                self._automap_p1.update(frame1)
            print(
                "[DEBUG] reset automap p0:",
                None if frame0 is None else getattr(frame0, "shape", None),
                "p1:",
                None if frame1 is None else getattr(frame1, "shape", None),
                flush=True,
            )
        if self.cfg.show_pov and self._pov is not None:
            if self.cfg.pov_view in ("p0", "p1", "both", "none"):
                self._view = self.cfg.pov_view
            frame0 = host_msg.get("screen")
            frame1 = join_msg.get("screen")
            if frame0 is not None:
                self._pov_frames[0] = frame0
            if frame1 is not None:
                self._pov_frames[1] = frame1
            show_pid = 0 if self._view in (None, "p0", "both") else 1
            frame = self._pov_frames.get(show_pid)
            if frame is not None:
                key = self._pov.update(frame)
                if key in (13, ord("v"), ord("V")):
                    self._toggle_view()
                    print("[ViZDoomMPProc] toggled view:", self._view, flush=True)

        # STEP 3: Finalize reset state and return the first observation.
        self._update_last_health(0, host_msg.get("obs", {}))
        self._update_last_health(1, join_msg.get("obs", {}))
        return {0: host_msg.get("obs", {}), 1: join_msg.get("obs", {})}

    def _score_from_obs(self, obs: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        frag = obs.get("FRAGCOUNT")
        score = obs.get("SCORE")
        total = obs.get("total_reward")
        return frag, score, total

    def _update_last_health(self, pid: int, obs: Dict[str, Any]) -> None:
        if "HEALTH" not in obs:
            return
        try:
            health = float(obs["HEALTH"])
            self._last_health[pid] = health
            if health > 0:
                self._seen_alive[pid] = True
        except (TypeError, ValueError):
            pass

    def _effective_health(self, pid: int, obs: Dict[str, Any]) -> Optional[float]:
        if "HEALTH" in obs:
            try:
                return float(obs["HEALTH"])
            except (TypeError, ValueError):
                return self._last_health.get(pid)
        return self._last_health.get(pid)

    def _is_dead(self, health: Optional[float]) -> bool:
        if health is None:
            return False
        try:
            return float(health) <= 0
        except (TypeError, ValueError):
            return False

    def _can_apply_death_done(self) -> bool:
        if int(self.t) <= int(self.cfg.death_check_warmup_steps):
            return False
        return bool(self._seen_alive.get(0, False) and self._seen_alive.get(1, False))

    def _compute_outcome(
        self,
        done: bool,
        obs0: Dict[str, Any],
        obs1: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Optional[float]]]:
        frag0, score0, total0 = self._score_from_obs(obs0)
        frag1, score1, total1 = self._score_from_obs(obs1)
        health0 = self._effective_health(0, obs0)
        health1 = self._effective_health(1, obs1)

        outcome = "draw"
        if done:
            if health0 is not None and health1 is not None:
                dead0 = self._is_dead(health0)
                dead1 = self._is_dead(health1)
                if dead0 and not dead1:
                    outcome = "p1_win"
                elif dead1 and not dead0:
                    outcome = "p0_win"
            elif frag0 is not None and frag1 is not None:
                if frag0 > frag1:
                    outcome = "p0_win"
                elif frag1 > frag0:
                    outcome = "p1_win"
            elif score0 is not None and score1 is not None:
                if score0 > score1:
                    outcome = "p0_win"
                elif score1 > score0:
                    outcome = "p1_win"
            elif total0 is not None and total1 is not None:
                if total0 > total1:
                    outcome = "p0_win"
                elif total1 > total0:
                    outcome = "p1_win"

        metrics: Dict[str, Optional[float]] = {
            "p0_frag": frag0,
            "p1_frag": frag1,
            "p0_score": score0,
            "p1_score": score1,
            "p0_total_reward": total0,
            "p1_total_reward": total1,
            "p0_health": health0,
            "p1_health": health1,
        }
        return outcome, metrics

    def _update_round_stats(self, done: bool, obs0: Dict[str, Any], obs1: Dict[str, Any]) -> None:
        if not done or self._round_done:
            return
        self._round_done = True
        self.match_count += 1
        frag0 = obs0.get("FRAGCOUNT")
        frag1 = obs1.get("FRAGCOUNT")
        health0 = self._effective_health(0, obs0)
        health1 = self._effective_health(1, obs1)
        winner: Optional[int] = None
        if health0 is not None and health1 is not None:
            dead0 = self._is_dead(health0)
            dead1 = self._is_dead(health1)
            if dead0 and not dead1:
                winner = 1
            elif dead1 and not dead0:
                winner = 0
        if winner is None and frag0 is not None and frag1 is not None:
            if frag0 > frag1:
                winner = 0
            elif frag1 > frag0:
                winner = 1
        if winner is not None:
            self.scores[winner] = self.scores.get(winner, 0) + 1
            print(
                f"[MATCH] Round {self.match_count}: Player {winner + 1} wins "
                f"(frag p1={frag0}, p2={frag1}).",
                flush=True,
            )
        else:
            print(
                f"[MATCH] Round {self.match_count}: Draw (frag p1={frag0}, p2={frag1}).",
                flush=True,
            )
        print(
            f"[MATCH] Total scores after {self.match_count} rounds: "
            f"Player 1: {self.scores.get(0, 0)} | Player 2: {self.scores.get(1, 0)}",
            flush=True,
        )

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, Any], Dict[int, float], bool, Dict]:
        assert self._host_conn is not None
        assert self._join_conn is not None

        self.t += 1
        a0 = int(actions.get(0, 0))
        a1 = int(actions.get(1, 0))

        self._host_conn.send({"type": "step", "action": a0, "t": self.t})
        self._join_conn.send({"type": "step", "action": a1, "t": self.t})

        host_msg = self._recv_with_timeout(self._host_conn, 20.0)
        join_msg = self._recv_with_timeout(self._join_conn, 20.0)
        if host_msg.get("type") == "error":
            raise RuntimeError(f"Host step failed: {host_msg.get('error')}")
        if join_msg.get("type") == "error":
            raise RuntimeError(f"Join step failed: {join_msg.get('error')}")

        obs0 = host_msg.get("obs", {})
        obs1 = join_msg.get("obs", {})
        self._update_last_health(0, obs0)
        self._update_last_health(1, obs1)
        r0 = float(host_msg.get("reward", 0.0))
        r1 = float(join_msg.get("reward", 0.0))
        done0 = bool(host_msg.get("done", False))
        done1 = bool(join_msg.get("done", False))
        death0 = self._is_dead(self._effective_health(0, obs0))
        death1 = self._is_dead(self._effective_health(1, obs1))
        death_done = (
            (death0 or death1)
            and not self.cfg.allow_respawn
            and self._can_apply_death_done()
        )
        done = done0 or done1 or death_done or (self.t >= self.cfg.max_steps)
        logger.info(
            "ViZDoomMPProc step t={} done0={} done1={} death0={} death1={} death_done={} obs0={} obs1={}",
            self.t,
            done0,
            done1,
            death0,
            death1,
            death_done,
            obs0,
            obs1,
        )

        outcome, metrics = self._compute_outcome(done, obs0, obs1)
        info: Dict[str, Any] = {"t": self.t, "outcome": outcome}
        info.update(metrics)
        self._update_round_stats(done, obs0, obs1)
        if self.cfg.show_automap:
            print(
                "[DEBUG step] automap p0=",
                None if host_msg.get("automap") is None else host_msg["automap"].shape,
                "p1=",
                None if join_msg.get("automap") is None else join_msg["automap"].shape,
                "t=", self.t,
                flush=True,
            )

        if self.cfg.show_automap:
            frame0 = host_msg.get("automap")
            frame1 = join_msg.get("automap")
            if self._automap_p0 is not None and frame0 is not None:
                key = self._automap_p0.update(frame0)
                if key in (13, ord("v"), ord("V")):
                    self._toggle_view()
                    print("[ViZDoomMPProc] toggled view:", self._view, flush=True)
            if self._automap_p1 is not None and frame1 is not None:
                self._automap_p1.update(frame1)
            if self.t % 30 == 0:
                print(
                    "[DEBUG] step automap p0:",
                    None if frame0 is None else getattr(frame0, "shape", None),
                    "p1:",
                    None if frame1 is None else getattr(frame1, "shape", None),
                    flush=True,
                )
        if self.cfg.show_pov and self._pov is not None:
            frame0 = host_msg.get("screen")
            frame1 = join_msg.get("screen")
            if frame0 is not None:
                self._pov_frames[0] = frame0
            if frame1 is not None:
                self._pov_frames[1] = frame1
            show_pid = 0 if self._view in (None, "p0", "both") else 1
            frame = self._pov_frames.get(show_pid)
            if frame is not None:
                key = self._pov.update(frame)
                if key in (13, ord("v"), ord("V")):
                    self._toggle_view()
                    print("[ViZDoomMPProc] toggled view:", self._view, flush=True)

        if self.cfg.render_mode is not None and self.cfg.sleep_s > 0:
            time.sleep(self.cfg.sleep_s)

        return {0: obs0, 1: obs1}, {0: r0, 1: r1}, done, info
