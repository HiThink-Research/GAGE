from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


_VISUAL_SESSION_LIFECYCLES = {"initializing", "live_running", "live_ended", "closed"}
_PLAYBACK_MODES = {"live_tail", "paused", "replay_playing"}
_OBSERVER_KINDS = {"global", "player", "spectator", "camera"}
_SCHEDULING_FAMILIES = {"turn", "agent_cycle", "record_cadence", "real_time_tick"}
_SCHEDULING_PHASES = {"idle", "waiting_for_intent", "advancing", "recording", "completed"}
_TIMELINE_TYPES = {
    "action_intent",
    "action_committed",
    "decision_window_open",
    "decision_window_close",
    "snapshot",
    "frame_ref",
    "chat",
    "system_marker",
    "result",
}
_TIMELINE_SEVERITIES = {"info", "warn", "critical"}
_SCENE_KINDS = {"board", "table", "frame", "rts"}
_SCENE_PHASES = {"live", "replay"}
_MEDIA_TRANSPORTS = {"artifact_ref", "http_pull", "binary_stream", "low_latency_channel"}
_RECEIPT_STATES = {"pending", "accepted", "committed", "rejected", "expired"}
_CONTROL_COMMAND_TYPES = {
    "follow_tail",
    "pause",
    "replay",
    "seek_seq",
    "seek_end",
    "step",
    "set_speed",
    "back_to_tail",
}
_CHAT_CHANNELS = {"table", "system", "private"}
_SNAPSHOT_MODES = {"full", "delta"}


def _jsonify(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


def _as_dict(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(payload or {})


def _as_string_tuple(payload: Sequence[Any] | None) -> tuple[str, ...]:
    if payload is None:
        return ()
    if isinstance(payload, (str, bytes)):
        raise TypeError("expected a sequence, not a string")
    return tuple(str(item) for item in payload)


def _as_sequence(payload: Any) -> Sequence[Any]:
    if payload is None:
        return ()
    if isinstance(payload, (str, bytes)):
        raise TypeError("expected a sequence, not a string")
    return payload


def _strict_optional_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in payload or payload.get(key) is None:
        return default
    value = payload[key]
    if isinstance(value, bool):
        return value
    raise TypeError(f"{key} must be a boolean or null")


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _validate_choice(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)!r}")
    return value


@dataclass(frozen=True, slots=True)
class PlaybackState:
    mode: str = "live_tail"
    cursor_ts: int = 0
    cursor_event_seq: int = 0
    speed: float = 1.0
    can_seek: bool = True

    def __post_init__(self) -> None:
        _validate_choice(self.mode, _PLAYBACK_MODES, "mode")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "cursorTs": self.cursor_ts,
            "cursorEventSeq": self.cursor_event_seq,
            "speed": self.speed,
            "canSeek": self.can_seek,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlaybackState":
        payload = payload or {}
        return cls(
            mode=str(payload.get("mode", "live_tail")),
            cursor_ts=int(payload.get("cursorTs", 0)),
            cursor_event_seq=int(payload.get("cursorEventSeq", 0)),
            speed=float(payload.get("speed", 1.0)),
            can_seek=_strict_optional_bool(payload, "canSeek", True),
        )


@dataclass(frozen=True, slots=True)
class ObserverRef:
    observer_id: str = ""
    observer_kind: str = "spectator"

    def __post_init__(self) -> None:
        _validate_choice(self.observer_kind, _OBSERVER_KINDS, "observer_kind")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observerId": self.observer_id,
            "observerKind": self.observer_kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObserverRef":
        payload = payload or {}
        return cls(
            observer_id=str(payload.get("observerId", "")),
            observer_kind=str(payload.get("observerKind", "spectator")),
        )


@dataclass(frozen=True, slots=True)
class ControlCommand:
    command_type: str
    target_seq: int | None = None
    step_delta: int | None = None
    speed: float | None = None
    issued_by: ObserverRef | None = None

    def __post_init__(self) -> None:
        _validate_choice(self.command_type, _CONTROL_COMMAND_TYPES, "command_type")
        if self.command_type in {"follow_tail", "pause", "replay", "seek_end", "back_to_tail"}:
            if self.target_seq is not None:
                raise ValueError(f"{self.command_type} does not allow target_seq")
            if self.step_delta is not None:
                raise ValueError(f"{self.command_type} does not allow step_delta")
            if self.speed is not None:
                raise ValueError(f"{self.command_type} does not allow speed")
        if self.command_type == "seek_seq" and self.target_seq is None:
            raise ValueError("seek_seq requires target_seq")
        if self.command_type == "seek_seq":
            if self.step_delta is not None:
                raise ValueError("seek_seq does not allow step_delta")
            if self.speed is not None:
                raise ValueError("seek_seq does not allow speed")
        if self.command_type == "step":
            if self.step_delta is None:
                raise ValueError("step requires step_delta")
            if self.step_delta not in {-1, 1}:
                raise ValueError("step_delta must be -1 or 1")
            if self.target_seq is not None:
                raise ValueError("step does not allow target_seq")
            if self.speed is not None:
                raise ValueError("step does not allow speed")
        if self.command_type == "set_speed":
            if self.speed is None:
                raise ValueError("set_speed requires speed")
            if self.speed <= 0:
                raise ValueError("speed must be positive")
            if self.target_seq is not None:
                raise ValueError("set_speed does not allow target_seq")
            if self.step_delta is not None:
                raise ValueError("set_speed does not allow step_delta")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "commandType": self.command_type,
        }
        if self.target_seq is not None:
            payload["targetSeq"] = self.target_seq
        if self.step_delta is not None:
            payload["stepDelta"] = self.step_delta
        if self.speed is not None:
            payload["speed"] = self.speed
        if self.issued_by is not None:
            payload["issuedBy"] = self.issued_by.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlCommand":
        payload = payload or {}
        issued_by_payload = payload.get("issuedBy")
        return cls(
            command_type=_require_string(payload, "commandType"),
            target_seq=None if payload.get("targetSeq") is None else int(payload["targetSeq"]),
            step_delta=None if payload.get("stepDelta") is None else int(payload["stepDelta"]),
            speed=None if payload.get("speed") is None else float(payload["speed"]),
            issued_by=(
                None if issued_by_payload is None else ObserverRef.from_dict(issued_by_payload)
            ),
        )


@dataclass(frozen=True, slots=True)
class ChatMessage:
    player_id: str
    text: str
    channel: str = "table"

    def __post_init__(self) -> None:
        _validate_choice(self.channel, _CHAT_CHANNELS, "channel")

    def to_dict(self) -> dict[str, Any]:
        return {
            "playerId": self.player_id,
            "text": self.text,
            "channel": self.channel,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChatMessage":
        payload = payload or {}
        return cls(
            player_id=_require_string(payload, "playerId"),
            text=_require_string(payload, "text"),
            channel=str(payload.get("channel", "table")),
        )


@dataclass(frozen=True, slots=True)
class SeekSnapshotRecord:
    seq: int
    ts_ms: int
    snapshot_mode: str
    snapshot_ref: str

    def __post_init__(self) -> None:
        _validate_choice(self.snapshot_mode, _SNAPSHOT_MODES, "snapshot_mode")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "tsMs": self.ts_ms,
            "snapshotMode": self.snapshot_mode,
            "snapshotRef": self.snapshot_ref,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SeekSnapshotRecord":
        payload = payload or {}
        return cls(
            seq=int(payload["seq"]),
            ts_ms=int(payload["tsMs"]),
            snapshot_mode=str(payload["snapshotMode"]),
            snapshot_ref=_require_string(payload, "snapshotRef"),
        )


@dataclass(frozen=True, slots=True)
class SchedulingState:
    family: str = "turn"
    phase: str = "idle"
    accepts_human_intent: bool = False
    active_actor_id: str | None = None
    window_id: str | None = None

    def __post_init__(self) -> None:
        _validate_choice(self.family, _SCHEDULING_FAMILIES, "family")
        _validate_choice(self.phase, _SCHEDULING_PHASES, "phase")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "family": self.family,
            "phase": self.phase,
            "acceptsHumanIntent": self.accepts_human_intent,
        }
        if self.active_actor_id is not None:
            payload["activeActorId"] = self.active_actor_id
        if self.window_id is not None:
            payload["windowId"] = self.window_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SchedulingState":
        payload = payload or {}
        return cls(
            family=str(payload.get("family", "turn")),
            phase=str(payload.get("phase", "idle")),
            accepts_human_intent=_strict_optional_bool(payload, "acceptsHumanIntent", False),
            active_actor_id=payload.get("activeActorId"),
            window_id=payload.get("windowId"),
        )


@dataclass(frozen=True, slots=True)
class MediaSourceRef:
    media_id: str
    transport: str
    mime_type: str | None = None
    url: str | None = None
    preview_ref: str | None = None

    def __post_init__(self) -> None:
        _validate_choice(self.transport, _MEDIA_TRANSPORTS, "transport")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "mediaId": self.media_id,
            "transport": self.transport,
        }
        if self.mime_type is not None:
            payload["mimeType"] = self.mime_type
        if self.url is not None:
            payload["url"] = self.url
        if self.preview_ref is not None:
            payload["previewRef"] = self.preview_ref
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MediaSourceRef":
        payload = payload or {}
        return cls(
            media_id=_require_string(payload, "mediaId"),
            transport=str(payload["transport"]),
            mime_type=payload.get("mimeType"),
            url=payload.get("url"),
            preview_ref=payload.get("previewRef"),
        )


@dataclass(frozen=True, slots=True)
class VisualSceneMedia:
    primary: MediaSourceRef | None = None
    auxiliary: tuple[MediaSourceRef, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.primary is not None:
            payload["primary"] = self.primary.to_dict()
        if self.auxiliary:
            payload["auxiliary"] = [media.to_dict() for media in self.auxiliary]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VisualSceneMedia":
        payload = payload or {}
        primary_payload = payload.get("primary")
        auxiliary_payload = _as_sequence(payload.get("auxiliary"))
        return cls(
            primary=None if primary_payload is None else MediaSourceRef.from_dict(primary_payload),
            auxiliary=tuple(MediaSourceRef.from_dict(item) for item in auxiliary_payload),
        )


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    seq: int
    ts_ms: int
    type: str
    label: str
    actor_id: str | None = None
    ref_snapshot_seq: int | None = None
    detail: str | None = None
    severity: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    payload: Any = None

    def __post_init__(self) -> None:
        _validate_choice(self.type, _TIMELINE_TYPES, "type")
        if self.severity is not None:
            _validate_choice(self.severity, _TIMELINE_SEVERITIES, "severity")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "seq": self.seq,
            "tsMs": self.ts_ms,
            "type": self.type,
            "label": self.label,
        }
        if self.actor_id is not None:
            payload["actorId"] = self.actor_id
        if self.ref_snapshot_seq is not None:
            payload["refSnapshotSeq"] = self.ref_snapshot_seq
        if self.detail is not None:
            payload["detail"] = self.detail
        if self.severity is not None:
            payload["severity"] = self.severity
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.payload is not None:
            payload["payload"] = _jsonify(self.payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TimelineEvent":
        payload = payload or {}
        return cls(
            seq=int(payload["seq"]),
            ts_ms=int(payload["tsMs"]),
            type=str(payload["type"]),
            label=str(payload["label"]),
            actor_id=payload.get("actorId"),
            ref_snapshot_seq=payload.get("refSnapshotSeq"),
            detail=payload.get("detail"),
            severity=payload.get("severity"),
            tags=_as_string_tuple(payload.get("tags")),
            payload=payload.get("payload"),
        )


@dataclass(frozen=True, slots=True)
class VisualSession:
    session_id: str
    game_id: str
    plugin_id: str
    lifecycle: str = "initializing"
    playback: PlaybackState = field(default_factory=PlaybackState)
    observer: ObserverRef = field(default_factory=ObserverRef)
    scheduling: SchedulingState = field(default_factory=SchedulingState)
    capabilities: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    timeline: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_choice(self.lifecycle, _VISUAL_SESSION_LIFECYCLES, "lifecycle")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "gameId": self.game_id,
            "pluginId": self.plugin_id,
            "lifecycle": self.lifecycle,
            "playback": self.playback.to_dict(),
            "observer": self.observer.to_dict(),
            "scheduling": self.scheduling.to_dict(),
            "capabilities": _jsonify(self.capabilities),
            "summary": _jsonify(self.summary),
            "timeline": _jsonify(self.timeline),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VisualSession":
        payload = payload or {}
        playback_payload = payload.get("playback", {})
        observer_payload = payload.get("observer", {})
        scheduling_payload = payload.get("scheduling", {})
        timeline_payload = _as_dict(payload.get("timeline"))
        return cls(
            session_id=_require_string(payload, "sessionId"),
            game_id=_require_string(payload, "gameId"),
            plugin_id=_require_string(payload, "pluginId"),
            lifecycle=str(payload.get("lifecycle", "initializing")),
            playback=PlaybackState.from_dict(playback_payload),
            observer=ObserverRef.from_dict(observer_payload),
            scheduling=SchedulingState.from_dict(scheduling_payload),
            capabilities=_as_dict(payload.get("capabilities")),
            summary=_as_dict(payload.get("summary")),
            timeline=_as_dict(timeline_payload),
        )


@dataclass(frozen=True, slots=True)
class VisualScene:
    scene_id: str
    game_id: str
    plugin_id: str
    kind: str
    ts_ms: int
    seq: int
    phase: str
    active_player_id: str | None
    legal_actions: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    summary: dict[str, Any] = field(default_factory=dict)
    body: Any = field(default_factory=dict)
    media: VisualSceneMedia | None = None
    overlays: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_choice(self.kind, _SCENE_KINDS, "kind")
        _validate_choice(self.phase, _SCENE_PHASES, "phase")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sceneId": self.scene_id,
            "gameId": self.game_id,
            "pluginId": self.plugin_id,
            "kind": self.kind,
            "tsMs": self.ts_ms,
            "seq": self.seq,
            "phase": self.phase,
            "activePlayerId": self.active_player_id,
            "legalActions": [_jsonify(action) for action in self.legal_actions],
            "summary": _jsonify(self.summary),
            "body": _jsonify(self.body),
        }
        if self.media is not None:
            media_payload = self.media.to_dict()
            if media_payload:
                payload["media"] = media_payload
        if self.overlays:
            payload["overlays"] = [_jsonify(overlay) for overlay in self.overlays]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VisualScene":
        payload = payload or {}
        media_payload = payload.get("media")
        overlays_payload = _as_sequence(payload.get("overlays"))
        legal_actions_payload = _as_sequence(payload.get("legalActions"))
        return cls(
            scene_id=_require_string(payload, "sceneId"),
            game_id=_require_string(payload, "gameId"),
            plugin_id=_require_string(payload, "pluginId"),
            kind=str(payload["kind"]),
            ts_ms=int(payload["tsMs"]),
            seq=int(payload["seq"]),
            phase=str(payload["phase"]),
            active_player_id=payload.get("activePlayerId"),
            legal_actions=tuple(
                _as_dict(action) if isinstance(action, Mapping) else action
                for action in legal_actions_payload
            ),
            summary=_as_dict(payload.get("summary")),
            body=payload.get("body"),
            media=None if media_payload is None else VisualSceneMedia.from_dict(media_payload),
            overlays=tuple(_as_dict(overlay) for overlay in overlays_payload),
        )


@dataclass(frozen=True, slots=True)
class ActionIntentReceipt:
    intent_id: str
    state: str = "pending"
    related_event_seq: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _validate_choice(self.state, _RECEIPT_STATES, "state")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "intentId": self.intent_id,
            "state": self.state,
        }
        if self.related_event_seq is not None:
            payload["relatedEventSeq"] = self.related_event_seq
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionIntentReceipt":
        return cls(
            intent_id=_require_string(payload, "intentId"),
            state=str(payload.get("state", "pending")),
            related_event_seq=payload.get("relatedEventSeq"),
            reason=payload.get("reason"),
        )
