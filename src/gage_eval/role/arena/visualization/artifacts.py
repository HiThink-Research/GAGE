from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from gage_eval.role.arena.visualization.contracts import TimelineEvent, VisualSession


def to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_json_safe(value.to_dict())
    if is_dataclass(value):
        return {
            field.name: to_json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_json_safe(item) for item in value]
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    if isinstance(value, set):
        return [to_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, frozenset):
        return [to_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [to_json_safe(item) for item in value]
    return str(value)


_HEAVY_KEY_NAMES = {
    "_rgb",
    "blob",
    "buffer",
    "frame",
    "frames",
    "image",
    "payload",
    "raw_frame",
    "raw_obs",
    "raw_observation",
    "rawobservation",
}
_VISUAL_HEAVY_KEY_NAMES = {
    "_rgb",
    "blob",
    "buffer",
    "raw_frame",
    "raw_obs",
    "raw_observation",
    "rawobservation",
    "rgb",
    "rgb_array",
    "frame_rgb",
}
_MAX_BOUNDED_DEPTH = 3
_MAX_BOUNDED_ITEMS = 8
_MAX_BOUNDED_STRING = 256
_MAX_VISUAL_DEPTH = 8
_MAX_VISUAL_ITEMS = 64
_MAX_VISUAL_STRING = 8192
_DATA_URL_KEYS = {"data_url", "dataurl", "url"}


def to_bounded_json_safe(
    value: Any,
    *,
    max_depth: int = _MAX_BOUNDED_DEPTH,
    max_items: int = _MAX_BOUNDED_ITEMS,
    max_string: int = _MAX_BOUNDED_STRING,
) -> Any:
    return _bounded_json_safe(
        value,
        depth=0,
        max_depth=max_depth,
        max_items=max_items,
        max_string=max_string,
        heavy_key_names=_HEAVY_KEY_NAMES,
    )


def to_visual_json_safe(
    value: Any,
    *,
    max_depth: int = _MAX_VISUAL_DEPTH,
    max_items: int = _MAX_VISUAL_ITEMS,
    max_string: int = _MAX_VISUAL_STRING,
) -> Any:
    return _bounded_json_safe(
        value,
        depth=0,
        max_depth=max_depth,
        max_items=max_items,
        max_string=max_string,
        heavy_key_names=_VISUAL_HEAVY_KEY_NAMES,
    )


def _bounded_json_safe(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    max_items: int,
    max_string: int,
    heavy_key_names: set[str],
) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return f"{value[:max_string]}...<len={len(value)}>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "kind": "bytes",
            "size": len(value),
        }
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _bounded_json_safe(
            value.to_dict(),
            depth=depth,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            heavy_key_names=heavy_key_names,
        )
    if is_dataclass(value):
        value = {
            field.name: getattr(value, field.name)
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return _bounded_mapping_snapshot(
            value,
            depth=depth,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            heavy_key_names=heavy_key_names,
        )
    if isinstance(value, tuple):
        return _bounded_sequence_snapshot(
            value,
            depth=depth,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            heavy_key_names=heavy_key_names,
        )
    if isinstance(value, list):
        return _bounded_sequence_snapshot(
            value,
            depth=depth,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            heavy_key_names=heavy_key_names,
        )
    if isinstance(value, set):
        return _bounded_sequence_snapshot(
            sorted(value, key=str),
            depth=depth,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            heavy_key_names=heavy_key_names,
        )
    if isinstance(value, frozenset):
        return _bounded_sequence_snapshot(
            sorted(value, key=str),
            depth=depth,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            heavy_key_names=heavy_key_names,
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _bounded_sequence_snapshot(
            list(value),
            depth=depth,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            heavy_key_names=heavy_key_names,
        )
    return str(value)


def _bounded_mapping_snapshot(
    payload: Mapping[str, Any],
    *,
    depth: int,
    max_depth: int,
    max_items: int,
    max_string: int,
    heavy_key_names: set[str],
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    contains_data_url = isinstance(payload.get("data_url"), str) or isinstance(payload.get("dataUrl"), str)
    for index, (key, item) in enumerate(payload.items()):
        if index >= max_items:
            snapshot["__truncated__"] = len(payload) - max_items
            break
        key_text = str(key)
        if _should_preserve_data_url(key_text, item):
            snapshot[key_text] = item
            continue
        if contains_data_url and key_text == "data":
            snapshot[key_text] = _summarize_heavy_value(item, max_items=max_items, max_string=max_string)
            continue
        if _is_heavy_key(key_text, heavy_key_names=heavy_key_names):
            snapshot[key_text] = _summarize_heavy_value(item, max_items=max_items, max_string=max_string)
            continue
        if depth >= max_depth:
            snapshot[key_text] = _summarize_heavy_value(item, max_items=max_items, max_string=max_string)
            continue
        snapshot[key_text] = _bounded_json_safe(
            item,
            depth=depth + 1,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            heavy_key_names=heavy_key_names,
        )
    return snapshot


def _bounded_sequence_snapshot(
    payload: Sequence[Any],
    *,
    depth: int,
    max_depth: int,
    max_items: int,
    max_string: int,
    heavy_key_names: set[str],
) -> list[Any]:
    snapshot: list[Any] = []
    for index, item in enumerate(payload):
        if index >= max_items:
            snapshot.append({"__truncated__": len(payload) - max_items})
            break
        if depth >= max_depth:
            snapshot.append(_summarize_heavy_value(item, max_items=max_items, max_string=max_string))
            continue
        snapshot.append(
            _bounded_json_safe(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
                heavy_key_names=heavy_key_names,
            )
        )
    return snapshot


def _summarize_heavy_value(
    value: Any,
    *,
    max_items: int,
    max_string: int,
) -> dict[str, Any]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"kind": "bytes", "size": len(value)}
    if isinstance(value, str):
        if len(value) <= max_string:
            return {"kind": "string", "value": value}
        return {"kind": "string", "size": len(value), "value": f"{value[:max_string]}...<len={len(value)}>"}
    if isinstance(value, Mapping):
        keys = [str(key) for key in list(value.keys())[:max_items]]
        return {"kind": "mapping", "size": len(value), "keys": keys}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {"kind": "sequence", "size": len(value)}
    if is_dataclass(value):
        return {"kind": type(value).__name__}
    return {"kind": type(value).__name__}


def _is_heavy_key(key: str, *, heavy_key_names: set[str]) -> bool:
    key_lower = key.lower()
    return key_lower in heavy_key_names


def _should_preserve_data_url(key: str, value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("data:")
        and key.lower() in _DATA_URL_KEYS
    )


@dataclass(frozen=True, slots=True)
class ArenaVisualArtifactLayout:
    replay_path: Path
    version: str = "v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "replay_path", Path(self.replay_path).expanduser())
        object.__setattr__(self, "version", str(self.version or "v1"))

    @classmethod
    def from_replay_path(cls, replay_path: str | Path, *, version: str = "v1") -> "ArenaVisualArtifactLayout":
        return cls(replay_path=Path(replay_path).expanduser(), version=version)

    @property
    def replay_dir(self) -> Path:
        return self.replay_path.parent

    @property
    def session_dir(self) -> Path:
        return self.replay_dir / "arena_visual_session" / self.version

    @property
    def timeline_path(self) -> Path:
        return self.session_dir / "timeline.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.session_dir / "manifest.json"

    @property
    def index_path(self) -> Path:
        return self.session_dir / "index.json"

    @property
    def seek_snapshots_path(self) -> Path:
        return self.session_dir / "seek_snapshots.json"

    @property
    def snapshot_dir(self) -> Path:
        return self.session_dir / "snapshots"

    @property
    def visual_session_ref(self) -> str:
        return str(self.manifest_path)

    def relative_ref(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.session_dir))
        except ValueError:
            return str(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "replayPath": str(self.replay_path),
            "replayDir": str(self.replay_dir),
            "sessionDir": str(self.session_dir),
            "timelineRef": str(self.timeline_path),
            "manifestRef": str(self.manifest_path),
            "indexRef": str(self.index_path),
            "seekSnapshotsRef": str(self.seek_snapshots_path),
            "snapshotDir": str(self.snapshot_dir),
            "visualSessionRef": self.visual_session_ref,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ArenaVisualSessionArtifacts:
    layout: ArenaVisualArtifactLayout
    visual_session: VisualSession
    timeline_events: tuple[TimelineEvent, ...] = ()
    snapshot_anchors: tuple[dict[str, Any], ...] = ()
    marker_index: dict[str, Any] = field(default_factory=dict)
    manifest_payload: dict[str, Any] = field(default_factory=dict)
    index_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeline_events", tuple(self.timeline_events))
        object.__setattr__(self, "snapshot_anchors", tuple(dict(anchor) for anchor in self.snapshot_anchors))
        object.__setattr__(self, "marker_index", dict(self.marker_index or {}))
        object.__setattr__(self, "manifest_payload", dict(self.manifest_payload or {}))
        object.__setattr__(self, "index_payload", dict(self.index_payload or {}))

    @property
    def manifest_path(self) -> Path:
        return self.layout.manifest_path

    @property
    def timeline_path(self) -> Path:
        return self.layout.timeline_path

    @property
    def index_path(self) -> Path:
        return self.layout.index_path

    @property
    def snapshot_dir(self) -> Path:
        return self.layout.snapshot_dir

    @property
    def seek_snapshots_path(self) -> Path:
        return self.layout.seek_snapshots_path

    @property
    def visual_session_ref(self) -> str:
        return self.layout.visual_session_ref

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(self.manifest_payload)


def _visual_session_summary(
    *,
    result: Any,
    event_count: int,
    snapshot_count: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "eventCount": int(event_count),
        "snapshotCount": int(snapshot_count),
    }
    if result is not None:
        summary["result"] = to_json_safe(result)
    return summary


def build_visual_session_manifest(
    *,
    layout: ArenaVisualArtifactLayout,
    visual_session: VisualSession,
    timeline_events: Sequence[TimelineEvent],
    snapshot_anchors: Sequence[Mapping[str, Any]],
    marker_index: Mapping[str, Any],
    result: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_snapshot_anchors = [
        {
            **dict(anchor),
            "snapshotRef": layout.relative_ref(Path(str(anchor["snapshotRef"]))),
        }
        if anchor.get("snapshotRef")
        else dict(anchor)
        for anchor in snapshot_anchors
    ]
    timeline_counts = {
        str(key): len(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else int(value)
        for key, value in marker_index.items()
    }
    timeline_manifest = {
        "eventCount": len(timeline_events),
        "headSeq": timeline_events[0].seq if timeline_events else 0,
        "tailSeq": timeline_events[-1].seq if timeline_events else 0,
        "markers": timeline_counts,
        "snapshotAnchors": normalized_snapshot_anchors,
        "timelineRef": layout.relative_ref(layout.timeline_path),
        "indexRef": layout.relative_ref(layout.index_path),
        "manifestRef": layout.relative_ref(layout.manifest_path),
    }
    session_payload = VisualSession(
        session_id=visual_session.session_id,
        game_id=visual_session.game_id,
        plugin_id=visual_session.plugin_id,
        lifecycle=visual_session.lifecycle,
        playback=visual_session.playback,
        observer=visual_session.observer,
        scheduling=visual_session.scheduling,
        capabilities=visual_session.capabilities,
        summary=_visual_session_summary(
            result=result,
            event_count=len(timeline_events),
            snapshot_count=len(snapshot_anchors),
        ),
        timeline=timeline_manifest,
    )
    manifest_payload = {
        "schema": "arena_visual_session/v1",
        "version": "1.0.0",
        "visualSession": session_payload.to_dict(),
        "timeline": timeline_manifest,
        "artifacts": {
            "timeline_ref": layout.relative_ref(layout.timeline_path),
            "index_ref": layout.relative_ref(layout.index_path),
            "seek_snapshots_ref": layout.relative_ref(layout.seek_snapshots_path),
            "manifest_ref": layout.relative_ref(layout.manifest_path),
            "snapshot_dir_ref": layout.relative_ref(layout.snapshot_dir),
            "visual_session_ref": layout.visual_session_ref,
            "snapshot_refs": [
                layout.relative_ref(Path(str(anchor.get("snapshotRef"))))
                for anchor in snapshot_anchors
                if anchor.get("snapshotRef")
            ],
        },
    }
    index_payload = {
        "schema": "arena_visual_session/v1",
        "version": "1.0.0",
        "timelineRef": layout.relative_ref(layout.timeline_path),
        "manifestRef": layout.relative_ref(layout.manifest_path),
        "eventCount": len(timeline_events),
        "markers": {
            str(key): list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [int(value)]
            for key, value in marker_index.items()
        },
        "snapshotAnchors": normalized_snapshot_anchors,
    }
    return manifest_payload, index_payload
