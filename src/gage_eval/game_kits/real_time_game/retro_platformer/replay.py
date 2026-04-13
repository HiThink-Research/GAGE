"""Replay schema writer for retro platformer environments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gage_eval.role.arena.replay_paths import resolve_replay_manifest_path
from gage_eval.role.arena.types import ArenaAction, GameResult


@dataclass
class ReplayFrame:
    """Stores a replay frame reference."""

    tick: int
    frame_path: Optional[str]
    shape: Optional[tuple[int, ...]]
    dtype: Optional[str]


class ReplaySchemaWriter:
    """Collects tick data and writes a GAGE replay schema."""

    def __init__(
        self,
        *,
        game: str,
        state: Optional[str],
        run_id: Optional[str],
        sample_id: Optional[str],
        replay_output_dir: Optional[str],
        replay_filename: Optional[str],
        frame_output_dir: Optional[str],
        frame_stride: int,
        snapshot_stride: int,
        rom_path: Optional[str],
    ) -> None:
        self._game = str(game)
        self._state = str(state) if state else None
        self._run_id = str(run_id) if run_id else None
        self._sample_id = str(sample_id) if sample_id else None
        self._replay_output_dir = str(replay_output_dir) if replay_output_dir else None
        self._replay_filename = str(replay_filename) if replay_filename else None
        self._frame_output_dir = str(frame_output_dir) if frame_output_dir else None
        self._frame_stride = max(1, int(frame_stride))
        self._snapshot_stride = max(1, int(snapshot_stride))
        self._rom_path = str(rom_path) if rom_path else None
        self._moves: list[dict[str, Any]] = []
        self._frames: list[dict[str, Any]] = []
        self._snapshots: list[dict[str, Any]] = []
        self._frame_index = 0
        self._replay_path: Optional[str] = None

    def append_decision(
        self,
        action: ArenaAction,
        *,
        decision_index: int,
        start_tick: int,
        end_tick: int,
    ) -> None:
        metadata = dict(action.metadata or {})
        self._moves.append(
            {
                "decision_index": decision_index,
                "player": action.player,
                "move": action.move,
                "raw": action.raw,
                "start_tick": start_tick,
                "end_tick": end_tick,
                "hold_ticks": metadata.get("hold_ticks"),
                "error": metadata.get("error"),
                "latency_ms": metadata.get("latency_ms"),
                "timed_out": metadata.get("timed_out"),
                "fallback_used": metadata.get("fallback_used"),
                "llm_wait_mode": metadata.get("llm_wait_mode"),
            }
        )

    def append_tick(
        self,
        *,
        tick: int,
        reward: float,
        info: dict[str, Any],
        frame: Optional[np.ndarray],
        done: bool,
    ) -> None:
        if tick % self._snapshot_stride == 0:
            self._snapshots.append({"tick": tick, "reward": reward, "info": info, "done": done})
        if frame is not None and tick % self._frame_stride == 0:
            frame_entry = self._store_frame(tick, frame)
            self._frames.append(frame_entry)

    def finalize(self, result: GameResult) -> Optional[str]:
        output_path = self._resolve_replay_output_path()
        if output_path is None:
            return None
        resolved_status = str(getattr(result, "status", None) or result.result)
        scores = getattr(result, "scores", None)
        metrics = getattr(result, "metrics", None)
        replay_payload = {
            "meta": {
                "game": self._game,
                "state": self._state,
                "rom_path": self._rom_path,
                "run_id": self._run_id,
                "sample_id": self._sample_id,
            },
            "moves": list(self._moves),
            "frames": list(self._frames),
            "snapshots": list(self._snapshots),
            "result": {
                "winner": result.winner,
                "status": resolved_status,
                "reason": result.reason,
                "move_count": result.move_count,
                "illegal_move_count": result.illegal_move_count,
                "replay_path": result.replay_path,
                "scores": dict(scores or {}) if isinstance(scores, dict) else {},
                "metrics": dict(metrics or {}) if isinstance(metrics, dict) else {},
            },
        }
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(replay_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return None
        self._replay_path = str(output_path)
        return self._replay_path

    def _store_frame(self, tick: int, frame: np.ndarray) -> dict[str, Any]:
        frame_path: Optional[str] = None
        shape: Optional[tuple[int, ...]] = None
        dtype: Optional[str] = None
        if isinstance(frame, np.ndarray):
            shape = tuple(frame.shape)
            dtype = str(frame.dtype)
        if self._frame_output_dir:
            output_dir = Path(self._frame_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"frame_{self._frame_index:06d}_tick_{tick}.npy"
            output_path = output_dir / filename
            np.save(output_path, frame)
            frame_path = str(output_path)
            self._frame_index += 1
        return {
            "tick": tick,
            "frame_path": frame_path,
            "shape": shape,
            "dtype": dtype,
        }

    def _resolve_replay_output_path(self) -> Optional[Path]:
        return resolve_replay_manifest_path(
            run_id=self._run_id,
            sample_id=self._sample_id,
            output_dir=self._replay_output_dir,
        )


__all__ = ["ReplayFrame", "ReplaySchemaWriter"]
