from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import os
import random
import time

try:
    import vizdoom  # type: ignore
    from vizdoom import DoomGame, GameVariable, Mode  # type: ignore
    _VIZDOOM_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on local optional deps.
    vizdoom = None  # type: ignore[assignment]
    DoomGame = GameVariable = Mode = Any  # type: ignore[assignment]
    _VIZDOOM_IMPORT_ERROR = exc


@dataclass
class ViZDoomMPEnvConfig:
    config_path: Optional[str] = None
    render_mode: Optional[str] = None  # None | "p0" | "p1" | "both"
    max_steps: int = 500
    action_repeat: int = 1
    sleep_s: float = 0.0
    port: Optional[int] = None


class ViZDoomMPEnv:
    """
    Minimal multiplayer ViZDoom env adapter (2 players, localhost host/join).
    - reset(seed) -> {pid: obs}
    - step({pid: action_int}) -> (obs_by_pid, rewards_by_pid, done, info)
    """
    def __init__(self, config: ViZDoomMPEnvConfig):
        self.cfg = config
        self.game_host: Optional[DoomGame] = None
        self.game_join: Optional[DoomGame] = None
        self.t = 0
        self.port: Optional[int] = None

        self.players = (0, 1)
        self._action_map: Dict[int, List[int]] = {}
        self._game_vars: List[GameVariable] = []
        self._map_name = "map01"
        self._has_frag = False
        self._has_score = False
        self._last_health: Dict[int, Optional[float]] = {0: None, 1: None}

    def _resolve_cfg_path(self) -> str:
        _ensure_vizdoom_available()
        if self.cfg.config_path:
            p = self.cfg.config_path
            if os.path.isabs(p):
                return p
            return os.path.abspath(p)

        pkg_dir = os.path.dirname(vizdoom.__file__)
        # return os.path.join(pkg_dir, "scenarios", "multi_duel.cfg")
        return os.path.join(pkg_dir, "scenarios", "timidity.cfg")

    def _parse_map_name(self, cfg_path: str) -> str:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip().lower()
                    if key in ("doom_map", "map"):
                        return val.strip().strip('"').strip("'")
        except Exception:
            pass
        return "map01"

    def _build_action_map(self, game: DoomGame) -> Dict[int, List[int]]:
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

    def _select_port(self, attempt: int) -> int:
        if attempt == 0 and self.cfg.port is not None:
            return int(self.cfg.port)
        return random.randint(5200, 5900)

    def _init_games(self) -> None:
        _ensure_vizdoom_available()
        cfg_path = self._resolve_cfg_path()
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"ViZDoom cfg not found: {cfg_path}")

        self._map_name = self._parse_map_name(cfg_path)

        last_err: Optional[Exception] = None
        for attempt in range(5):
            port = self._select_port(attempt)
            host_args = f"-host 2 -port {port} -deathmatch +timelimit 1 +sv_spawnfarthest 1 +viz_connect_timeout 120"
            join_args = f"-join 127.0.0.1 -port {port}"
            print(f"[ViZDoomMP] cfg={cfg_path} port={port}")
            print(f"[ViZDoomMP] host_args='{host_args}'")
            print(f"[ViZDoomMP] join_args='{join_args}'")

            host = DoomGame()
            join = DoomGame()
            try:
                host.load_config(cfg_path)
                join.load_config(cfg_path)

                host.add_game_args(host_args)
                join.add_game_args(join_args)

                host.set_mode(Mode.ASYNC_PLAYER)
                join.set_mode(Mode.ASYNC_PLAYER)

                if hasattr(GameVariable, "HEALTH"):
                    host.add_available_game_variable(GameVariable.HEALTH)
                    join.add_available_game_variable(GameVariable.HEALTH)

                rm = self.cfg.render_mode
                host.set_window_visible(rm in ("p0", "both"))
                join.set_window_visible(rm in ("p1", "both"))

                host.init()
                # Give the host time to open the server socket before join connects.
                time.sleep(2.0)
                join.init()

                # Allow the join to finish syncing before starting the episode.
                time.sleep(2.0)
                host.new_episode()
                join.new_episode()

                self.game_host = host
                self.game_join = join
                self.port = port

                self._action_map = self._build_action_map(host)
                self._game_vars = list(host.get_available_game_variables())
                var_names = {v.name for v in self._game_vars}
                self._has_frag = "FRAGCOUNT" in var_names
                self._has_score = "SCORE" in var_names
                return
            except Exception as e:
                last_err = e
                try:
                    host.close()
                except Exception:
                    pass
                try:
                    join.close()
                except Exception:
                    pass
                time.sleep(0.1)

        raise RuntimeError(f"Failed to init multiplayer ViZDoom after retries: {last_err}")

    def close(self) -> None:
        if self.game_host is not None:
            try:
                self.game_host.close()
            except Exception:
                pass
        if self.game_join is not None:
            try:
                self.game_join.close()
            except Exception:
                pass
        self.game_host = None
        self.game_join = None

    def reset(self, seed: Optional[int] = None) -> Dict[int, Any]:
        if self.game_host is None or self.game_join is None:
            self._init_games()

        assert self.game_host is not None
        assert self.game_join is not None

        if seed is not None:
            try:
                self.game_host.set_seed(int(seed))
                self.game_join.set_seed(int(seed))
            except Exception:
                pass

        self.game_host.new_episode()
        self.game_join.new_episode()
        self.t = 0
        self._last_health = {0: None, 1: None}

        obs0 = self._get_obs(self.game_host)
        obs1 = self._get_obs(self.game_join)
        self._update_last_health(0, obs0)
        self._update_last_health(1, obs1)
        return {0: obs0, 1: obs1}

    def _get_obs(self, game: DoomGame) -> Dict[str, Any]:
        obs: Dict[str, Any] = {"t": self.t}
        try:
            obs["total_reward"] = float(game.get_total_reward())
        except Exception:
            pass

        for var in self._game_vars:
            try:
                obs[var.name] = float(game.get_game_variable(var))
            except Exception:
                pass

        return obs

    def _get_score_values(self, game: DoomGame) -> Tuple[Optional[float], Optional[float]]:
        frag = None
        score = None
        if self._has_frag:
            try:
                frag = float(game.get_game_variable(GameVariable.FRAGCOUNT))
            except Exception:
                frag = None
        if self._has_score:
            try:
                score = float(game.get_game_variable(GameVariable.SCORE))
            except Exception:
                score = None
        return frag, score

    def _get_health_values(self) -> Tuple[Optional[float], Optional[float]]:
        if self.game_host is None or self.game_join is None:
            return None, None
        health0 = None
        health1 = None
        try:
            health0 = float(self.game_host.get_game_variable(GameVariable.HEALTH))
        except Exception:
            health0 = None
        try:
            health1 = float(self.game_join.get_game_variable(GameVariable.HEALTH))
        except Exception:
            health1 = None
        return health0, health1

    def _update_last_health(self, pid: int, obs: Dict[str, Any]) -> None:
        if "HEALTH" not in obs:
            return
        try:
            self._last_health[pid] = float(obs["HEALTH"])
        except (TypeError, ValueError):
            pass

    def _effective_health(self, pid: int, health: Optional[float]) -> Optional[float]:
        if health is None:
            return self._last_health.get(pid)
        return health

    def _is_dead(self, health: Optional[float]) -> bool:
        if health is None:
            return False
        try:
            return float(health) <= 0
        except (TypeError, ValueError):
            return False

    def _compute_outcome(self, done: bool) -> Tuple[str, Dict[str, Optional[float]]]:
        assert self.game_host is not None
        assert self.game_join is not None

        frag0, score0 = self._get_score_values(self.game_host)
        frag1, score1 = self._get_score_values(self.game_join)
        raw_health0, raw_health1 = self._get_health_values()
        health0 = self._effective_health(0, raw_health0)
        health1 = self._effective_health(1, raw_health1)

        total0 = None
        total1 = None
        try:
            total0 = float(self.game_host.get_total_reward())
            total1 = float(self.game_join.get_total_reward())
        except Exception:
            pass

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

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, Any], Dict[int, float], bool, Dict]:
        assert self.game_host is not None
        assert self.game_join is not None

        self.t += 1

        done0 = self.game_host.is_episode_finished()
        done1 = self.game_join.is_episode_finished()
        raw_health0, raw_health1 = self._get_health_values()
        health0 = self._effective_health(0, raw_health0)
        health1 = self._effective_health(1, raw_health1)
        death_done = self._is_dead(health0) or self._is_dead(health1)
        if done0 or done1 or death_done:
            outcome, metrics = self._compute_outcome(True)
            info: Dict[str, Any] = {"t": self.t, "outcome": outcome}
            info.update(metrics)
            obs0 = self._get_obs(self.game_host)
            obs1 = self._get_obs(self.game_join)
            self._update_last_health(0, obs0)
            self._update_last_health(1, obs1)
            return {0: obs0, 1: obs1}, {0: 0.0, 1: 0.0}, True, info

        a0 = int(actions.get(0, 0))
        a1 = int(actions.get(1, 0))
        vec0 = self._action_map.get(a0, self._action_map[0])
        vec1 = self._action_map.get(a1, self._action_map[0])

        r0 = float(self.game_host.make_action(vec0, self.cfg.action_repeat))
        r1 = float(self.game_join.make_action(vec1, self.cfg.action_repeat))

        done0 = self.game_host.is_episode_finished()
        done1 = self.game_join.is_episode_finished()
        raw_health0, raw_health1 = self._get_health_values()
        health0 = self._effective_health(0, raw_health0)
        health1 = self._effective_health(1, raw_health1)
        death_done = self._is_dead(health0) or self._is_dead(health1)
        done = done0 or done1 or death_done or (self.t >= self.cfg.max_steps)

        outcome, metrics = self._compute_outcome(done)
        info: Dict[str, Any] = {"t": self.t, "outcome": outcome}
        info.update(metrics)

        if self.cfg.render_mode is not None and self.cfg.sleep_s > 0:
            time.sleep(self.cfg.sleep_s)

        obs0 = self._get_obs(self.game_host)
        obs1 = self._get_obs(self.game_join)
        self._update_last_health(0, obs0)
        self._update_last_health(1, obs1)
        rewards = {0: r0, 1: r1}

        return {0: obs0, 1: obs1}, rewards, done, info


def _ensure_vizdoom_available() -> None:
    if vizdoom is not None:
        return
    raise RuntimeError(
        "ViZDoom backend is unavailable because optional dependency 'vizdoom' is not installed."
    ) from _VIZDOOM_IMPORT_ERROR
