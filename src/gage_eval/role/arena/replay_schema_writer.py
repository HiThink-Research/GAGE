"""Unified replay writer for gage_replay/v1 artifacts."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gage_eval.role.arena.types import GameResult


class ReplaySchemaWriter:
    """Writes replay.json and events.jsonl for a single sample."""

    def __init__(self, *, run_dir: Path, sample_id: str, output_dir: Optional[str] = None) -> None:
        """Initialize replay writer paths.

        Args:
            run_dir: Run directory (`runs/<run_id>`).
            sample_id: Sample id used for replay subdirectory naming.
            output_dir: Optional custom replay root directory.
        """

        self._run_dir = Path(run_dir).expanduser().resolve()
        self._sample_id = str(sample_id or "unknown")
        self._output_dir = str(output_dir) if output_dir else None

    def write(
        self,
        *,
        scheduler_type: str,
        result: GameResult,
        move_log: Sequence[dict[str, Any]],
        arena_trace: Optional[Sequence[Mapping[str, Any]]],
        extra_meta: Optional[dict[str, Any]] = None,
        recording_mode: Optional[str] = None,
    ) -> str:
        """Write replay schema artifacts and return replay.json path.

        Args:
            scheduler_type: Arena scheduler kind.
            result: Final game result payload.
            move_log: Action history for event conversion.
            arena_trace: Optional ordered scheduler trace entries.
            extra_meta: Optional metadata passed from adapter/environment.
            recording_mode: Optional replay recording mode (`action`/`frame`/`both`).

        Returns:
            Absolute path to the generated replay.json file.

        Raises:
            RuntimeError: If output files cannot be written.
        """

        replay_dir = self._resolve_replay_dir()
        events_path = replay_dir / "events.jsonl"
        replay_path = replay_dir / "replay.json"
        meta_extra = dict(extra_meta or {})
        requested_mode = _normalize_recording_mode(recording_mode)

        # STEP 1: Convert move_log entries to action events.
        action_events = self._build_action_events(move_log) if _mode_includes_action(requested_mode) else []

        # STEP 2: Convert optional frame events and append result event.
        frame_events = self._build_frame_events(
            meta_extra.pop("frame_events", None),
            start_seq=len(action_events) + 1,
        )
        if not _mode_includes_frame(requested_mode):
            frame_events = []
        result_event = self._build_result_event(
            seq=len(action_events) + len(frame_events) + 1,
            result=result,
        )
        events = [*action_events, *frame_events, result_event]

        # STEP 3: Build replay manifest.
        replay_payload = self._build_replay_payload(
            scheduler_type=scheduler_type,
            result=result,
            action_count=len(action_events),
            frame_count=len(frame_events),
            arena_trace=arena_trace,
            extra_meta=meta_extra,
            recording_mode=requested_mode,
        )

        # STEP 4: Persist events and manifest.
        replay_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._write_events(events_path, events)
            replay_path.write_text(
                json.dumps(replay_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - defensive filesystem guard
            raise RuntimeError(f"failed_to_write_replay_v1:{exc}") from exc
        return str(replay_path.resolve())

    def _resolve_replay_dir(self) -> Path:
        return self.resolve_replay_dir(
            run_dir=self._run_dir,
            sample_id=self._sample_id,
            output_dir=self._output_dir,
        )

    @staticmethod
    def resolve_replay_dir(
        *,
        run_dir: Path,
        sample_id: str,
        output_dir: Optional[str] = None,
    ) -> Path:
        """Resolve replay directory from run/sample identifiers.

        Args:
            run_dir: Run directory (`runs/<run_id>`).
            sample_id: Raw sample identifier.
            output_dir: Optional replay output root override.

        Returns:
            Absolute replay directory path for this sample.
        """

        safe_sample_id = _sanitize_sample_id(sample_id)
        if output_dir:
            base_dir = Path(output_dir).expanduser()
            if not base_dir.is_absolute():
                base_dir = (Path.cwd() / base_dir).resolve()
            return base_dir / safe_sample_id
        return Path(run_dir).expanduser().resolve() / "replays" / safe_sample_id

    def _build_action_events(self, move_log: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for index, raw_entry in enumerate(move_log, start=1):
            entry = dict(raw_entry or {})
            timestamp_ms = _resolve_timestamp_ms(entry)
            step_value = _resolve_step(entry, index)
            actor = _resolve_actor(entry)
            move = _resolve_move(entry)
            raw = _resolve_raw(entry, move)
            meta = _extract_meta(entry)
            event = {
                "type": "action",
                "seq": index,
                "ts_ms": timestamp_ms,
                "step": step_value,
                "actor": actor,
                "move": move,
                "raw": raw,
                "meta": meta,
            }
            events.append(event)
        return events

    def _build_frame_events(self, frame_events: Any, *, start_seq: int) -> list[dict[str, Any]]:
        if not isinstance(frame_events, Sequence):
            return []
        events: list[dict[str, Any]] = []
        for item in frame_events:
            if not isinstance(item, Mapping):
                continue
            frame = dict(item)
            frame.setdefault("type", "frame")
            frame["seq"] = int(start_seq) + len(events)
            frame.setdefault("ts_ms", _now_ms())
            events.append(frame)
        return events

    def _build_result_event(self, *, seq: int, result: GameResult) -> dict[str, Any]:
        return {
            "type": "result",
            "seq": int(seq),
            "ts_ms": _now_ms(),
            "winner": result.winner,
            "result": _resolve_result_status(result),
            "reason": result.reason,
        }

    def _build_replay_payload(
        self,
        *,
        scheduler_type: str,
        result: GameResult,
        action_count: int,
        frame_count: int,
        arena_trace: Optional[Sequence[Mapping[str, Any]]],
        extra_meta: dict[str, Any],
        recording_mode: Optional[str],
    ) -> dict[str, Any]:
        resolved_mode = recording_mode or _resolve_recording_mode(action_count, frame_count)
        payload: dict[str, Any] = {
            "schema": "gage_replay/v1",
            "version": "1.0.0",
            "meta": {
                "run_id": self._run_dir.name,
                "sample_id": self._sample_id,
                "scheduler_type": str(scheduler_type or "turn"),
                **extra_meta,
            },
            "recording": {
                "mode": resolved_mode,
                "events_path": "events.jsonl",
                "counts": {
                    "action": int(action_count),
                    "frame": int(frame_count),
                    "result": 1,
                },
            },
            "result": {
                "winner": result.winner,
                "result": _resolve_result_status(result),
                "reason": result.reason,
            },
            "stats": {
                "move_count": int(result.move_count),
                "illegal_move_count": int(result.illegal_move_count),
            },
        }
        normalized_trace = _normalize_arena_trace_steps(arena_trace)
        if normalized_trace:
            payload["arena_trace"] = normalized_trace
        visual_session_ref = extra_meta.get("visual_session_ref")
        if visual_session_ref not in (None, ""):
            payload["artifacts"] = {
                "visual_session_ref": str(visual_session_ref),
            }
        legacy_replay_path = payload["meta"].pop("legacy_replay_path", None)
        if legacy_replay_path:
            payload["files"] = {"legacy_replay_path": str(legacy_replay_path)}
        return payload

    @staticmethod
    def _write_events(path: Path, events: Sequence[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False, default=str))
                handle.write(os.linesep)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sanitize_sample_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return cleaned or "unknown"


def _normalize_arena_trace_steps(raw_trace: Any) -> list[dict[str, Any]]:
    trace_source = raw_trace
    if isinstance(trace_source, Mapping):
        # NOTE: Keep compatibility with legacy {"schema": "...", "steps": [...]} payloads.
        legacy_steps = trace_source.get("steps")
        if isinstance(legacy_steps, Sequence) and not isinstance(legacy_steps, (str, bytes)):
            trace_source = legacy_steps
        else:
            return []
    if not isinstance(trace_source, Sequence) or isinstance(trace_source, (str, bytes)):
        return []
    return [dict(item) for item in trace_source if isinstance(item, Mapping)]


def update_replay_manifest_visual_session_ref(*, replay_path: str | Path, visual_session_ref: str) -> bool:
    path = Path(replay_path).expanduser()
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifacts = payload.setdefault("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
            payload["artifacts"] = artifacts
        artifacts["visual_session_ref"] = str(visual_session_ref)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return True
    except Exception:
        return False


def _resolve_timestamp_ms(entry: Mapping[str, Any]) -> int:
    for key in ("timestamp_ms", "ts_ms", "timestamp", "time_ms", "time"):
        value = entry.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return _now_ms()


def _resolve_step(entry: Mapping[str, Any], fallback_index: int) -> int:
    for key in ("step", "index", "decision_index", "turn", "tick"):
        value = entry.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(fallback_index)


def _resolve_actor(entry: Mapping[str, Any]) -> str:
    for key in ("actor", "player", "player_id", "playerId"):
        value = entry.get(key)
        if value:
            return str(value)
    return "unknown"


def _resolve_move(entry: Mapping[str, Any]) -> str:
    for key in ("move", "action_text", "action", "action_cards", "action_id", "coord"):
        value = entry.get(key)
        if value is None:
            continue
        return str(value)
    return ""


def _resolve_raw(entry: Mapping[str, Any], move: str) -> str:
    raw = entry.get("raw")
    if raw is not None:
        return str(raw)
    return str(move)


def _extract_meta(entry: Mapping[str, Any]) -> dict[str, Any]:
    known_keys = {
        "type",
        "seq",
        "ts_ms",
        "timestamp_ms",
        "timestamp",
        "time_ms",
        "time",
        "step",
        "index",
        "decision_index",
        "turn",
        "tick",
        "actor",
        "player",
        "player_id",
        "playerId",
        "move",
        "action_text",
        "action",
        "action_cards",
        "action_id",
        "coord",
        "raw",
    }
    metadata = {
        str(key): _normalize_replay_json_value(value)
        for key, value in entry.items()
        if key not in known_keys
    }
    return metadata


def _normalize_replay_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _normalize_replay_json_value(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _normalize_replay_json_value(tolist())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _normalize_replay_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize_replay_json_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_replay_json_value(item) for item in value]
    return value


def _resolve_recording_mode(action_count: int, frame_count: int) -> str:
    if action_count > 0 and frame_count > 0:
        return "both"
    if frame_count > 0:
        return "frame"
    return "action"


def _normalize_recording_mode(value: Any) -> Optional[str]:
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode in {"action", "frame", "both"}:
        return mode
    return None


def _mode_includes_action(mode: Optional[str]) -> bool:
    if mode is None:
        return True
    return mode in {"action", "both"}


def _mode_includes_frame(mode: Optional[str]) -> bool:
    if mode is None:
        return True
    return mode in {"frame", "both"}


def _resolve_result_status(result: GameResult) -> str:
    status = getattr(result, "status", None)
    if status:
        return str(status)
    return str(result.result)
