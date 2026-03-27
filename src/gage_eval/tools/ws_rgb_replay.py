"""Unified ws_rgb replay service entrypoint."""

from __future__ import annotations

import argparse
import json
import signal
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from gage_eval.tools.ws_rgb_server import DisplayRegistration, WsRgbHubServer


class ReplayBuilder(Protocol):
    """Build a ws_rgb display payload for one sample replay."""

    def __call__(
        self,
        sample_record: Mapping[str, Any],
        *,
        task_id: str,
        fps: float,
        max_frames: int,
    ) -> dict[str, Any]:
        """Build and return replay display registration fields."""


@dataclass(frozen=True)
class WsRgbReplayOptions:
    """Runtime options for ws_rgb replay server."""

    host: str
    port: int
    fps: float
    max_frames: int
    game: Optional[str]
    auto_open: bool


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

        # STEP 1: Start replay cursor at first pull to avoid skipping to tail
        # when browser attaches after server startup.
        now = time.monotonic()
        if self._started_at is None:
            self._started_at = now
            self._last_advance_at = now
        elif self._index < len(self._frames) - 1:
            last_advance_at = self._last_advance_at if self._last_advance_at is not None else now
            elapsed_since_advance = max(0.0, now - last_advance_at)
            # STEP 2: Advance at most one frame per pull.
            # NOTE: This prevents frame skipping when poll interval is slower than fps.
            if elapsed_since_advance >= self._frame_interval_s:
                self._index += 1
                self._last_advance_at = now

        started_at = self._started_at if self._started_at is not None else now
        elapsed_s = max(0.0, now - started_at)
        return self._materialize_payload(index=self._index, elapsed_s=elapsed_s)

    def frame_count(self) -> int:
        """Return total replay frame count."""

        return len(self._frames)

    def frame_at(self, index: int) -> dict[str, Any]:
        """Return one replay frame by index without advancing cursor."""

        if not self._frames:
            return self._materialize_payload(index=0, elapsed_s=0.0)
        clamped = max(0, min(len(self._frames) - 1, int(index)))
        elapsed_s = max(0.0, float(clamped) / self._fps)
        return self._materialize_payload(index=clamped, elapsed_s=elapsed_s)

    def _materialize_payload(self, *, index: int, elapsed_s: float) -> dict[str, Any]:
        payload = dict(self._frames[index])
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = dict(metadata)
        metadata["replay_elapsed_s"] = max(0.0, float(elapsed_s))
        metadata["replay_index"] = int(index)
        metadata["replay_total"] = len(self._frames)
        payload["metadata"] = metadata
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve post-run replay through ws_rgb viewer.")
    parser.add_argument("--sample-json", required=True, help="Path to one sample JSON under runs/<run_id>/samples.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for ws_rgb server.")
    parser.add_argument("--port", type=int, default=5800, help="Bind port for ws_rgb server.")
    parser.add_argument("--fps", type=float, default=12.0, help="Replay playback fps for frame cursor.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional upper bound for reconstructed replay frames (0 means no limit).",
    )
    parser.add_argument(
        "--game",
        choices=("pettingzoo",),
        default=None,
        help="Optional explicit game key. When omitted, inferred from task metadata.",
    )
    parser.add_argument(
        "--auto-open",
        type=int,
        choices=(0, 1),
        default=0,
        help="Auto open browser viewer URL after server startup (0/1).",
    )
    return parser.parse_args()


def _load_sample_record(sample_json_path: str) -> tuple[dict[str, Any], str]:
    path = Path(sample_json_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"sample_json_not_found:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sample_json_invalid_root")
    payload["_gage_sample_json_path"] = str(path)
    sample = payload.get("sample")
    sample_task_id = ""
    if isinstance(sample, Mapping):
        sample_task_id = str(sample.get("task_id") or "")
    task_id = str(payload.get("task_id") or sample_task_id or "")
    return payload, task_id


def _infer_game_key(sample_record: Mapping[str, Any], *, explicit_game: Optional[str], task_id: str) -> str:
    if explicit_game:
        return str(explicit_game).strip().lower()

    normalized_task_id = str(task_id or "").strip().lower()
    if normalized_task_id.startswith("pettingzoo_"):
        return "pettingzoo"

    sample = sample_record.get("sample")
    if isinstance(sample, Mapping):
        metadata = sample.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("env_impl", "env_id", "pettingzoo_env_id", "game_id"):
                value = metadata.get(key)
                if value and "pettingzoo" in str(value).lower():
                    return "pettingzoo"
    raise ValueError("cannot_infer_game_key; pass --game explicitly")


def _resolve_builder(game_key: str) -> ReplayBuilder:
    normalized = str(game_key).strip().lower()
    if normalized == "pettingzoo":
        from gage_eval.role.arena.games.pettingzoo.ws_rgb_replay import (
            build_ws_rgb_replay_display,
        )

        return build_ws_rgb_replay_display
    raise ValueError(f"unsupported_game_key:{normalized}")


def _resolve_replay_manifest_path(sample_record: Mapping[str, Any]) -> Optional[Path]:
    sample = sample_record.get("sample")
    if not isinstance(sample, Mapping):
        return None
    predict_result = sample.get("predict_result")
    if not isinstance(predict_result, Sequence):
        return None
    for item in predict_result:
        if not isinstance(item, Mapping):
            continue
        for replay_ref in _iter_replay_manifest_refs(item):
            path = _resolve_replay_ref_path(replay_ref, sample_record=sample_record)
            if path is not None:
                return path
    return None


def _iter_replay_manifest_refs(entry: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("replay_path", "replay_v1_path"):
        value = entry.get(key)
        if value not in (None, ""):
            candidates.append(str(value))

    result = entry.get("result")
    if isinstance(result, Mapping):
        for key in ("replay_path", "replay_v1_path"):
            value = result.get(key)
            if value not in (None, ""):
                candidates.append(str(value))

    artifacts = entry.get("artifacts")
    if isinstance(artifacts, Mapping):
        for key in ("replay_ref", "replay_v1_ref"):
            value = artifacts.get(key)
            if value not in (None, ""):
                candidates.append(str(value))
    return candidates


def _resolve_replay_ref_path(
    replay_ref: str,
    *,
    sample_record: Mapping[str, Any],
) -> Optional[Path]:
    raw_path = Path(str(replay_ref)).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((Path.cwd() / raw_path).resolve())
        sample_json_path = sample_record.get("_gage_sample_json_path")
        if sample_json_path:
            sample_path = Path(str(sample_json_path)).expanduser().resolve()
            candidates.append((sample_path.parent / raw_path).resolve())
            for base in sample_path.parents[:4]:
                candidates.append((base / raw_path).resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_events_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except Exception:
            continue
        if isinstance(payload, Mapping):
            events.append(dict(payload))
    return events


def _trim_leading_empty_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Trim leading replay frames that do not carry visible content."""

    normalized = [dict(item) for item in frames if isinstance(item, Mapping)]
    if len(normalized) <= 1:
        return normalized

    trimmed = list(normalized)
    while len(trimmed) > 1 and not _has_visible_frame_content(trimmed[0]):
        trimmed.pop(0)
    return trimmed


def _has_visible_frame_content(frame_payload: Mapping[str, Any]) -> bool:
    """Return whether one replay frame contains user-visible content."""

    board_text = frame_payload.get("board_text")
    if isinstance(board_text, str) and board_text.strip():
        return True

    if _frame_has_image_path(frame_payload):
        return True

    for key in ("public_state", "private_state", "ui_state", "state", "observation", "legal_moves", "legal_actions"):
        if _is_non_empty_value(frame_payload.get(key)):
            return True

    ignored_keys = {
        "metadata",
        "active_player_id",
        "observer_player_id",
        "player_id",
        "player_ids",
        "player_names",
        "move_count",
        "last_move",
        "reward",
        "termination",
        "truncation",
        "env_id",
        "agent_id",
    }
    for key, value in frame_payload.items():
        if str(key) in ignored_keys:
            continue
        if _is_non_empty_value(value):
            return True
    return False


def _is_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) > 0
    return bool(value)


def _frame_has_image_path(frame_payload: Mapping[str, Any]) -> bool:
    for key in ("_image_path_abs", "image_path", "frame_image_path"):
        path = frame_payload.get(key)
        if path:
            return True
    image = frame_payload.get("image")
    if isinstance(image, Mapping) and image.get("path"):
        return True
    return False


def _build_replay_v1_display(
    sample_record: Mapping[str, Any],
    *,
    task_id: str,
    fps: float,
    max_frames: int,
) -> Optional[dict[str, Any]]:
    """Build ws_rgb display directly from replay v1 frame events."""

    replay_manifest_path = _resolve_replay_manifest_path(sample_record)
    if replay_manifest_path is None:
        return None
    try:
        replay_payload = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(replay_payload, Mapping):
        return None
    if replay_payload.get("schema") != "gage_replay/v1":
        return None
    recording = replay_payload.get("recording")
    if not isinstance(recording, Mapping):
        return None
    events_rel = str(recording.get("events_path") or "events.jsonl")
    events_path = (replay_manifest_path.parent / events_rel).resolve()
    if not events_path.exists():
        return None

    # STEP 1: Load frame events from replay event stream.
    frame_events = [
        event
        for event in _load_events_jsonl(events_path)
        if str(event.get("type") or "").strip().lower() == "frame"
    ]
    if not frame_events:
        return None
    if max_frames > 0:
        frame_events = frame_events[: max(0, int(max_frames))]

    # STEP 2: Build frame payloads consumed by ws_rgb frame_source.
    frames: list[dict[str, Any]] = []
    for event in frame_events:
        frame_payload_raw = event.get("frame")
        frame_payload = dict(frame_payload_raw) if isinstance(frame_payload_raw, Mapping) else {}
        frame_payload.setdefault("board_text", "")
        frame_payload.setdefault("move_count", event.get("step"))
        metadata = frame_payload.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        metadata = dict(metadata)
        metadata["replay_seq"] = event.get("seq")
        metadata["replay_step"] = event.get("step")
        frame_payload["metadata"] = metadata

        image = event.get("image")
        if isinstance(image, Mapping):
            image_path = image.get("path")
            if image_path:
                image_file = Path(str(image_path)).expanduser()
                if not image_file.is_absolute():
                    image_file = (replay_manifest_path.parent / image_file).resolve()
                frame_payload["_image_path_abs"] = str(image_file)
                frame_payload["image"] = {"path": str(image.get("path"))}
        frames.append(frame_payload)

    frames = _trim_leading_empty_frames(frames)
    cursor = ReplayFrameCursor(frames, fps=float(fps))

    # STEP 3: Assemble display registration metadata.
    sample = sample_record.get("sample")
    sample_id = "sample"
    human_player_id = "player_0"
    if isinstance(sample, Mapping):
        sample_id = str(sample.get("id") or "sample")
        metadata = sample.get("metadata")
        if isinstance(metadata, Mapping):
            player_ids = metadata.get("player_ids")
            if isinstance(player_ids, Sequence) and not isinstance(player_ids, (str, bytes)) and player_ids:
                human_player_id = str(player_ids[0])
    replay_name = replay_manifest_path.parent.name
    replay_meta = replay_payload.get("meta")
    scheduler_type = replay_meta.get("scheduler_type") if isinstance(replay_meta, Mapping) else None
    replay_label = str(task_id or scheduler_type or "replay_v1")
    return {
        "display_id": f"replay:{sample_id}:{replay_name}",
        "label": f"replay_v1:{replay_label}",
        "human_player_id": human_player_id,
        "frame_source": cursor.frame_source,
        "frame_at": cursor.frame_at,
        "frame_count": cursor.frame_count,
    }


def _serve_replay(
    *,
    sample_record: Mapping[str, Any],
    task_id: str,
    options: WsRgbReplayOptions,
) -> None:
    """Start ws_rgb replay service from one sample record."""

    replay_display = _build_replay_v1_display(
        sample_record,
        task_id=task_id,
        fps=float(options.fps),
        max_frames=max(0, int(options.max_frames)),
    )
    game_key = ""
    if replay_display is None:
        game_key = _infer_game_key(sample_record, explicit_game=options.game, task_id=task_id)
        builder = _resolve_builder(game_key)
        replay_display = builder(
            sample_record,
            task_id=task_id,
            fps=float(options.fps),
            max_frames=max(0, int(options.max_frames)),
        )

    frame_source = replay_display.get("frame_source")
    if not callable(frame_source):
        raise ValueError("replay_builder_missing_frame_source")
    frame_at = replay_display.get("frame_at")
    frame_count = replay_display.get("frame_count")

    hub = WsRgbHubServer(host=str(options.host), port=int(options.port), allow_origin="*")
    hub.start()
    hub.register_display(
        DisplayRegistration(
            display_id=str(replay_display.get("display_id") or f"replay:{game_key}"),
            label=str(replay_display.get("label") or f"{game_key}_replay"),
            human_player_id=str(replay_display.get("human_player_id") or "player_0"),
            frame_source=frame_source,
            frame_at=frame_at if callable(frame_at) else None,
            frame_count=frame_count if callable(frame_count) else None,
            input_mapper=None,
            action_queue=None,
        )
    )
    _maybe_open_browser(hub.viewer_url, enabled=options.auto_open)

    stop_flag = {"stop": False}

    def _handle_stop(_signum: int, _frame: Any) -> None:
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    print(f"ws_rgb replay ready: {hub.viewer_url}")
    print(f"display_id: {replay_display.get('display_id')}")
    print("Press Ctrl+C to stop replay server.")
    try:
        while not stop_flag["stop"]:
            time.sleep(0.25)
    finally:
        hub.stop()


def main() -> None:
    """Parse arguments and run ws_rgb replay service."""

    args = _parse_args()
    sample_record, task_id = _load_sample_record(args.sample_json)
    options = WsRgbReplayOptions(
        host=str(args.host),
        port=int(args.port),
        fps=float(args.fps),
        max_frames=max(0, int(args.max_frames)),
        game=str(args.game) if args.game else None,
        auto_open=bool(int(args.auto_open)),
    )
    _serve_replay(sample_record=sample_record, task_id=task_id, options=options)


def _maybe_open_browser(viewer_url: str, *, enabled: bool) -> None:
    """Opens viewer URL in default browser when auto-open is enabled."""

    if not enabled:
        return
    try:
        webbrowser.open(viewer_url)
    except Exception:
        return


if __name__ == "__main__":
    main()
