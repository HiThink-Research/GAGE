from __future__ import annotations

import base64
import binascii
import io
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from threading import Condition, RLock
import time
from typing import Callable, Protocol
from urllib.parse import quote, unquote, unquote_to_bytes

from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.role.arena.visualization.assembly import (
    assemble_visual_scene,
    collect_scene_media_refs,
)
from gage_eval.role.arena.visualization.contracts import (
    ControlCommand,
    MediaSourceRef,
    ObserverRef,
    TimelineEvent,
    VisualSceneMedia,
    VisualScene,
    VisualSession,
)
from gage_eval.role.arena.visualization.gateway_service import TimelinePage
from gage_eval.role.arena.visualization.recorder import (
    ArenaVisualSessionRecorder,
)

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

_SUPPORTED_LIVE_SCENE_SCHEMES = {
    "http_pull",
    "binary_stream",
    "low_latency_channel",
}
_BINARY_MEDIA_PREFIX = "live-binary:"
_LOW_LATENCY_MEDIA_PREFIX = "live-channel-"
_LOW_LATENCY_SCENE_KINDS = {"frame", "rts"}


class ArenaVisualLiveSessionSource(Protocol):
    session_id: str
    run_id: str | None

    def load_session(self, *, observer: ObserverRef | None = None) -> VisualSession: ...
    def load_live_header(self) -> Mapping[str, object]: ...
    def current_live_revision(self) -> int: ...
    def wait_for_live_revision(self, *, after_revision: int, timeout_s: float | None = None) -> int: ...

    def page_timeline(self, *, after_seq: int | None = None, limit: int = 50) -> TimelinePage: ...

    def load_scene(
        self,
        *,
        seq: int,
        observer: ObserverRef | None = None,
    ) -> VisualScene | None: ...

    def lookup_marker(self, marker: str) -> tuple[int, ...]: ...

    def lookup_media(self, media_id: str) -> MediaSourceRef | None: ...

    def load_media_content(self, media_id: str) -> tuple[bytes, str | None] | None: ...

    def load_stream_frame(self, media_id: str) -> tuple[bytes, str | None] | None: ...

    def apply_control_command(self, command: ControlCommand) -> int: ...


class ArenaVisualFinishGate:
    def __init__(self, *, idle_timeout_s: float) -> None:
        self._idle_timeout_s = max(0.0, float(idle_timeout_s))
        self._condition = Condition()
        self._armed = False
        self._manual_finish_required = False
        self._finished = False
        self._deadline_monotonic = 0.0

    def arm(self) -> None:
        with self._condition:
            if self._armed:
                return
            self._armed = True
            if self._idle_timeout_s <= 0.0:
                self._finished = True
            else:
                self._deadline_monotonic = time.monotonic() + self._idle_timeout_s
            self._condition.notify_all()

    def wait(self) -> None:
        with self._condition:
            while True:
                if not self._armed or self._finished:
                    return
                if self._manual_finish_required:
                    self._condition.wait()
                    continue
                remaining_s = self._deadline_monotonic - time.monotonic()
                if remaining_s <= 0.0:
                    self._finished = True
                    return
                self._condition.wait(timeout=remaining_s)

    def record_control_interaction(self) -> bool:
        with self._condition:
            if not self._armed or self._finished:
                return False
            self._manual_finish_required = True
            self._condition.notify_all()
            return True

    def finish(self) -> bool:
        with self._condition:
            if self._finished:
                return False
            self._finished = True
            self._manual_finish_required = False
            self._condition.notify_all()
            return True


@dataclass(frozen=True, slots=True)
class RecorderLiveSessionSource:
    recorder: ArenaVisualSessionRecorder
    run_id: str | None = None
    visualization_spec: GameVisualizationSpec | None = None
    live_scene_scheme: str = "http_pull"
    finish_gate: ArenaVisualFinishGate | None = None
    live_frame_supplier: Callable[[], object | None] | None = None
    stop_callback: Callable[[], object | None] | None = None
    restart_callback: Callable[[], object | None] | None = None
    _cache_lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)
    _session_cache: dict[tuple[int, str, str], VisualSession] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _live_state_cache: tuple[int, object] | None = field(default=None, init=False, repr=False, compare=False)
    _scene_cache: OrderedDict[tuple[int, int, str, str], VisualScene] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
        compare=False,
    )
    _stream_frame_cache: dict[str, tuple[int, tuple[bytes, str | None]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.live_scene_scheme not in _SUPPORTED_LIVE_SCENE_SCHEMES:
            raise ValueError(
                f"live_scene_scheme must be one of {sorted(_SUPPORTED_LIVE_SCENE_SCHEMES)!r}"
            )

    @property
    def session_id(self) -> str:
        return str(self.recorder.session_id)

    def load_session(self, *, observer: ObserverRef | None = None) -> VisualSession:
        revision = self.current_live_revision()
        observer_kind, observer_id = self._normalize_observer_key(observer)
        with self._cache_lock:
            cached = self._session_cache.get((revision, observer_kind, observer_id))
        if cached is not None:
            return cached
        session = self.recorder.build_live_session()
        adapted = _override_observer(session, observer=observer)
        with self._cache_lock:
            self._session_cache[(revision, observer_kind, observer_id)] = adapted
        return adapted

    def load_live_header(self) -> Mapping[str, object]:
        return self.recorder.export_live_header()

    def current_live_revision(self) -> int:
        current_revision = getattr(self.recorder, "current_live_revision", None)
        if callable(current_revision):
            return int(current_revision())
        return 0

    def wait_for_live_revision(self, *, after_revision: int, timeout_s: float | None = None) -> int:
        wait_for_revision = getattr(self.recorder, "wait_for_live_revision", None)
        if callable(wait_for_revision):
            return int(wait_for_revision(after_revision=after_revision, timeout_s=timeout_s))
        if timeout_s not in (None, 0):
            time.sleep(max(0.0, float(timeout_s)))
        return self.current_live_revision()

    def page_timeline(self, *, after_seq: int | None = None, limit: int = 50) -> TimelinePage:
        if limit <= 0:
            raise ValueError("limit must be positive")
        page_events, next_after_seq, has_more = self.recorder.page_live_timeline(
            after_seq=after_seq,
            limit=limit,
        )
        return TimelinePage(
            events=page_events,
            after_seq=after_seq,
            next_after_seq=next_after_seq,
            limit=limit,
            has_more=has_more,
        )

    def load_scene(
        self,
        *,
        seq: int,
        observer: ObserverRef | None = None,
    ) -> VisualScene | None:
        revision = self.current_live_revision()
        observer_kind, observer_id = self._normalize_observer_key(observer)
        cache_key = (revision, int(seq), observer_kind, observer_id)
        with self._cache_lock:
            cached_scene = self._scene_cache.get(cache_key)
            if cached_scene is not None:
                self._scene_cache.move_to_end(cache_key)
                return cached_scene
        state = self._load_live_state(revision=revision)
        raw_scene = self._load_raw_scene(state=state, seq=seq, observer=observer)
        if raw_scene is None:
            return None
        adapted_scene = self._adapt_scene_for_live_transport(raw_scene)
        with self._cache_lock:
            self._scene_cache[cache_key] = adapted_scene
            while len(self._scene_cache) > 16:
                self._scene_cache.popitem(last=False)
        return adapted_scene

    def lookup_marker(self, marker: str) -> tuple[int, ...]:
        return self.recorder.lookup_marker(marker)

    def lookup_media(self, media_id: str) -> MediaSourceRef | None:
        state = self._load_live_state(revision=self.current_live_revision())
        if self.live_scene_scheme == "binary_stream":
            binary_ref = self._lookup_binary_media_ref(state=state, media_id=media_id)
            if binary_ref is not None:
                return replace(binary_ref, media_id=media_id, transport="binary_stream", url=None)
        if self.live_scene_scheme == "low_latency_channel":
            channel_ref = self._lookup_low_latency_media_ref(state=state, media_id=media_id)
            if channel_ref is not None:
                return channel_ref
        raw_ref = self._lookup_raw_media_ref(state=state, media_id=media_id)
        if raw_ref is None:
            return None
        return self._adapt_media_ref(
            raw_ref=raw_ref,
            scene_kind="frame",
            scene_seq=0,
            stream_id=None,
            is_primary=False,
        )

    def load_media_content(self, media_id: str) -> tuple[bytes, str | None] | None:
        state = self._load_live_state(revision=self.current_live_revision())
        raw_ref = self._resolve_media_content_ref(state=state, media_id=media_id)
        if raw_ref is None:
            return None
        url = str(raw_ref.url or "").strip()
        if not url.startswith("data:"):
            return None
        return _decode_data_url(url)

    def load_stream_frame(self, media_id: str) -> tuple[bytes, str | None] | None:
        cached_stream_frame = self._load_cached_stream_frame(media_id)
        if cached_stream_frame is not None:
            return cached_stream_frame
        live_payload = self._load_live_frame_stream_payload(media_id)
        if live_payload is not None:
            content, mime_type = live_payload
            encoded = _encode_low_latency_frame(content=content, mime_type=mime_type)
            self._store_cached_stream_frame(media_id, encoded)
            return encoded
        payload = self.load_media_content(media_id)
        if payload is None:
            return None
        content, mime_type = payload
        encoded = _encode_low_latency_frame(content=content, mime_type=mime_type)
        self._store_cached_stream_frame(media_id, encoded)
        return encoded

    def apply_control_command(self, command: ControlCommand) -> int:
        if command.command_type == "finish":
            if self.stop_callback is not None:
                self.stop_callback()
            if self.finish_gate is not None:
                self.finish_gate.finish()
            return int(self.recorder.export_live_header().get("cursorEventSeq", 0))
        if command.command_type == "restart":
            if self.restart_callback is not None:
                result = self.restart_callback()
                if result is not None:
                    return int(result)
            return int(self.recorder.export_live_header().get("cursorEventSeq", 0))
        if self.finish_gate is not None:
            self.finish_gate.record_control_interaction()
        return self.recorder.apply_control_command(command)

    def _normalize_observer_key(self, observer: ObserverRef | None) -> tuple[str, str]:
        if observer is None:
            return str(self.recorder.observer_kind), str(self.recorder.observer_id)
        return str(observer.observer_kind), str(observer.observer_id)

    def _load_live_state(self, *, revision: int):
        with self._cache_lock:
            cached = self._live_state_cache
            if cached is not None and int(cached[0]) == int(revision):
                return cached[1]
        state = self.recorder.export_live_state()
        with self._cache_lock:
            if revision == self.current_live_revision():
                retained_sessions = {
                    key: value
                    for key, value in self._session_cache.items()
                    if key[0] == revision
                }
                retained_scenes = OrderedDict(
                    (key, value)
                    for key, value in self._scene_cache.items()
                    if key[0] == revision
                )
                self._session_cache.clear()
                self._session_cache.update(retained_sessions)
                self._scene_cache.clear()
                self._scene_cache.update(retained_scenes)
                object.__setattr__(self, "_live_state_cache", (revision, state))
        return state

    def _current_stream_frame_version(self) -> int | None:
        if self.live_frame_supplier is None:
            return None
        try:
            payload = self.live_frame_supplier()
        except Exception:
            return None
        if not isinstance(payload, Mapping):
            return None
        version = payload.get("_live_frame_version")
        try:
            return int(version)
        except (TypeError, ValueError):
            return id(payload)

    def _load_cached_stream_frame(self, media_id: str) -> tuple[bytes, str | None] | None:
        frame_version = self._current_stream_frame_version()
        if frame_version is None:
            return None
        with self._cache_lock:
            cached = self._stream_frame_cache.get(str(media_id))
        if cached is None:
            return None
        cached_version, cached_payload = cached
        if cached_version != frame_version:
            return None
        return cached_payload

    def _store_cached_stream_frame(
        self,
        media_id: str,
        payload: tuple[bytes, str | None],
    ) -> None:
        frame_version = self._current_stream_frame_version()
        if frame_version is None:
            return
        with self._cache_lock:
            self._stream_frame_cache[str(media_id)] = (frame_version, payload)

    def _load_raw_scene(
        self,
        *,
        state,
        seq: int,
        observer: ObserverRef | None = None,
    ) -> VisualScene | None:
        event = _lookup_event(state.timeline_events, seq=seq)
        if event is None:
            return None

        visual_session = _override_observer(state.visual_session, observer=observer)
        snapshot_anchor = _select_snapshot_anchor(state.snapshot_payloads, seq=seq)
        snapshot_body = None if snapshot_anchor is None else snapshot_anchor.get("snapshot")
        assembled = assemble_visual_scene(
            visual_session=visual_session,
            event=event,
            snapshot_anchor=(
                None
                if snapshot_anchor is None
                else {
                    "seq": int(snapshot_anchor["seq"]),
                    "tsMs": int(snapshot_anchor["tsMs"]),
                    "label": snapshot_anchor.get("label"),
                }
            ),
            snapshot_body=snapshot_body,
            visualization_spec=self.visualization_spec,
        )
        return replace(assembled, phase="live")

    def _adapt_scene_for_live_transport(self, scene: VisualScene) -> VisualScene:
        stream_id = _read_scene_stream_id(scene)
        if scene.media is None:
            synthetic_primary = self._build_synthetic_low_latency_media_ref(
                scene_kind=scene.kind,
                stream_id=stream_id,
            )
            if synthetic_primary is None:
                return scene
            return replace(scene, media=VisualSceneMedia(primary=synthetic_primary, auxiliary=()))
        primary = None
        if scene.media.primary is not None:
            primary = self._adapt_media_ref(
                raw_ref=scene.media.primary,
                scene_kind=scene.kind,
                scene_seq=scene.seq,
                stream_id=stream_id,
                is_primary=True,
            )
        auxiliary = tuple(
            ref
            for ref in (
                self._adapt_media_ref(
                    raw_ref=item,
                    scene_kind=scene.kind,
                    scene_seq=scene.seq,
                    stream_id=stream_id,
                    is_primary=False,
                )
                for item in scene.media.auxiliary
            )
            if ref is not None
        )
        if primary is None:
            primary = self._build_synthetic_low_latency_media_ref(
                scene_kind=scene.kind,
                stream_id=stream_id,
            )
        return replace(scene, media=VisualSceneMedia(primary=primary, auxiliary=auxiliary))

    def _adapt_media_ref(
        self,
        *,
        raw_ref: MediaSourceRef,
        scene_kind: str,
        scene_seq: int,
        stream_id: str | None,
        is_primary: bool,
    ) -> MediaSourceRef | None:
        url = str(raw_ref.url or "").strip()
        if self.live_scene_scheme == "http_pull" or not url.startswith("data:"):
            return raw_ref
        if self.live_scene_scheme == "binary_stream":
            return replace(
                raw_ref,
                media_id=self._build_binary_media_id(scene_seq=scene_seq, raw_media_id=raw_ref.media_id),
                transport="binary_stream",
                url=None,
            )
        if (
            self.live_scene_scheme == "low_latency_channel"
            and scene_kind in _LOW_LATENCY_SCENE_KINDS
            and is_primary
        ):
            channel_id = self._build_low_latency_media_id(stream_id or raw_ref.media_id)
            return MediaSourceRef(
                media_id=channel_id,
                transport="low_latency_channel",
                mime_type="multipart/x-mixed-replace",
                url=self._build_low_latency_stream_url(channel_id),
                preview_ref=raw_ref.preview_ref,
            )
        return replace(
            raw_ref,
            media_id=self._build_binary_media_id(scene_seq=scene_seq, raw_media_id=raw_ref.media_id),
            transport="binary_stream",
            url=None,
        )

    def _lookup_binary_media_ref(self, *, state, media_id: str) -> MediaSourceRef | None:
        parsed = _parse_binary_media_id(media_id)
        if parsed is None:
            return None
        scene_seq, raw_media_id = parsed
        scene = self._load_raw_scene(state=state, seq=scene_seq)
        if scene is None:
            return None
        refs = collect_scene_media_refs(scene)
        return refs.get(raw_media_id)

    def _lookup_raw_media_ref(self, *, state, media_id: str) -> MediaSourceRef | None:
        for event in reversed(state.timeline_events):
            scene = self._load_raw_scene(state=state, seq=event.seq)
            if scene is None:
                continue
            refs = collect_scene_media_refs(scene)
            ref = refs.get(media_id)
            if ref is not None:
                return ref
        return None

    def _lookup_low_latency_media_ref(self, *, state, media_id: str) -> MediaSourceRef | None:
        for event in reversed(state.timeline_events):
            scene = self._load_raw_scene(state=state, seq=event.seq)
            if scene is None or scene.kind not in _LOW_LATENCY_SCENE_KINDS:
                continue
            adapted = self._resolve_low_latency_primary_media(scene)
            if adapted is not None and adapted.media_id == media_id:
                return adapted
        return None

    def _resolve_media_content_ref(self, *, state, media_id: str) -> MediaSourceRef | None:
        if self.live_scene_scheme == "binary_stream" and media_id.startswith(_BINARY_MEDIA_PREFIX):
            return self._lookup_binary_media_ref(state=state, media_id=media_id)
        if self.live_scene_scheme == "low_latency_channel" and media_id.startswith(_LOW_LATENCY_MEDIA_PREFIX):
            live_ref = self._resolve_live_frame_media_ref(media_id)
            if live_ref is not None:
                return live_ref
            for event in reversed(state.timeline_events):
                scene = self._load_raw_scene(state=state, seq=event.seq)
                if (
                    scene is None
                    or scene.kind not in _LOW_LATENCY_SCENE_KINDS
                    or scene.media is None
                    or scene.media.primary is None
                ):
                    continue
                stream_id = _read_scene_stream_id(scene) or scene.media.primary.media_id
                if self._build_low_latency_media_id(stream_id) == media_id:
                    return scene.media.primary
            return None
        return self._lookup_raw_media_ref(state=state, media_id=media_id)

    def _resolve_live_frame_media_ref(self, media_id: str) -> MediaSourceRef | None:
        payload = self._load_live_frame_payload(media_id)
        if payload is None:
            return None
        media_payload = payload.get("media")
        if not isinstance(media_payload, Mapping):
            return None
        primary_payload = media_payload.get("primary")
        if not isinstance(primary_payload, Mapping):
            return None
        try:
            return MediaSourceRef.from_dict(primary_payload)
        except Exception:
            return None

    def _load_live_frame_stream_payload(self, media_id: str) -> tuple[bytes, str | None] | None:
        payload = self._load_live_frame_payload(media_id)
        if payload is None:
            return None
        live_ref = self._resolve_live_ref_from_payload(payload)
        if live_ref is not None:
            resolved = _load_media_content_from_ref(live_ref)
            if resolved is not None:
                return resolved
        return _load_rgb_frame_payload(payload)

    def _load_live_frame_payload(self, media_id: str) -> Mapping[str, object] | None:
        if (
            self.live_scene_scheme != "low_latency_channel"
            or self.live_frame_supplier is None
            or not media_id.startswith(_LOW_LATENCY_MEDIA_PREFIX)
        ):
            return None
        try:
            payload = self.live_frame_supplier()
        except Exception:
            return None
        if not isinstance(payload, Mapping):
            return None
        stream_id = _read_live_frame_stream_id(payload)
        if self._build_low_latency_media_id(stream_id) != media_id:
            return None
        return payload

    def _resolve_live_ref_from_payload(self, payload: Mapping[str, object]) -> MediaSourceRef | None:
        media_payload = payload.get("media")
        if not isinstance(media_payload, Mapping):
            return None
        primary_payload = media_payload.get("primary")
        if not isinstance(primary_payload, Mapping):
            return None
        try:
            ref = MediaSourceRef.from_dict(primary_payload)
        except Exception:
            return None
        return ref

    def _build_binary_media_id(self, *, scene_seq: int, raw_media_id: str) -> str:
        normalized_raw_media_id = str(raw_media_id).strip() or "primary"
        encoded_raw_media_id = quote(normalized_raw_media_id, safe="")
        return f"{_BINARY_MEDIA_PREFIX}{int(scene_seq)}:{encoded_raw_media_id}"

    def _build_low_latency_media_id(self, stream_id: str) -> str:
        normalized_stream_id = str(stream_id).strip() or "primary"
        return f"{_LOW_LATENCY_MEDIA_PREFIX}{normalized_stream_id}"

    def _build_low_latency_stream_url(self, media_id: str) -> str:
        session_path = quote(self.session_id, safe="")
        media_path = quote(media_id, safe="")
        if self.run_id is None or not str(self.run_id).strip():
            return f"/arena_visual/sessions/{session_path}/media/{media_path}/stream"
        run_path = quote(str(self.run_id), safe="")
        return f"/arena_visual/sessions/{session_path}/media/{media_path}/stream?run_id={run_path}"

    def _resolve_low_latency_primary_media(self, scene: VisualScene) -> MediaSourceRef | None:
        stream_id = _read_scene_stream_id(scene)
        if scene.media is not None and scene.media.primary is not None:
            adapted = self._adapt_media_ref(
                raw_ref=scene.media.primary,
                scene_kind=scene.kind,
                scene_seq=scene.seq,
                stream_id=stream_id,
                is_primary=True,
            )
            if adapted is not None:
                return adapted
        return self._build_synthetic_low_latency_media_ref(
            scene_kind=scene.kind,
            stream_id=stream_id,
        )

    def _build_synthetic_low_latency_media_ref(
        self,
        *,
        scene_kind: str,
        stream_id: str | None,
    ) -> MediaSourceRef | None:
        if (
            self.live_scene_scheme != "low_latency_channel"
            or self.live_frame_supplier is None
            or scene_kind not in _LOW_LATENCY_SCENE_KINDS
        ):
            return None
        normalized_stream_id = str(stream_id or "").strip()
        if not normalized_stream_id:
            return None
        channel_id = self._build_low_latency_media_id(normalized_stream_id)
        return MediaSourceRef(
            media_id=channel_id,
            transport="low_latency_channel",
            mime_type="multipart/x-mixed-replace",
            url=self._build_low_latency_stream_url(channel_id),
        )


class ArenaVisualLiveRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sources: dict[tuple[str, str | None], ArenaVisualLiveSessionSource] = {}

    def register(self, source: ArenaVisualLiveSessionSource) -> None:
        key = (str(source.session_id), _normalize_run_id(source.run_id))
        with self._lock:
            self._sources[key] = source

    def unregister(self, *, session_id: str, run_id: str | None = None) -> None:
        key = (str(session_id), _normalize_run_id(run_id))
        with self._lock:
            self._sources.pop(key, None)

    def resolve(self, *, session_id: str, run_id: str | None = None) -> ArenaVisualLiveSessionSource | None:
        normalized_run_id = _normalize_run_id(run_id)
        with self._lock:
            if normalized_run_id is not None:
                return self._sources.get((str(session_id), normalized_run_id))
            matches = [
                source
                for (candidate_session_id, _candidate_run_id), source in self._sources.items()
                if candidate_session_id == str(session_id)
            ]
        if len(matches) == 1:
            return matches[0]
        return None


def _lookup_event(events: Sequence[TimelineEvent], *, seq: int) -> TimelineEvent | None:
    for event in events:
        if event.seq == int(seq):
            return event
    return None


def _select_snapshot_anchor(
    snapshot_payloads: Sequence[dict[str, object]],
    *,
    seq: int,
) -> dict[str, object] | None:
    selected: dict[str, object] | None = None
    for candidate in snapshot_payloads:
        candidate_seq = int(candidate.get("seq", -1))
        if candidate_seq > int(seq):
            break
        selected = candidate
    return selected


def _override_observer(
    visual_session: VisualSession,
    *,
    observer: ObserverRef | None,
) -> VisualSession:
    if observer is None:
        return visual_session
    return replace(visual_session, observer=observer)


def _normalize_run_id(run_id: str | None) -> str | None:
    if run_id is None:
        return None
    text = str(run_id).strip()
    return text or None


def _read_scene_stream_id(scene: VisualScene) -> str | None:
    body = scene.body
    if not isinstance(body, dict):
        return None
    frame = body.get("frame")
    if not isinstance(frame, dict):
        return None
    stream_id = frame.get("streamId")
    if not isinstance(stream_id, str):
        return None
    normalized = stream_id.strip()
    return normalized or None


def _read_live_frame_stream_id(payload: Mapping[str, object]) -> str:
    stream_id = payload.get("stream_id")
    if isinstance(stream_id, str) and stream_id.strip():
        return stream_id.strip()
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        nested_stream_id = metadata.get("stream_id")
        if isinstance(nested_stream_id, str) and nested_stream_id.strip():
            return nested_stream_id.strip()
    return "main"


def _parse_binary_media_id(media_id: str) -> tuple[int, str] | None:
    if not media_id.startswith(_BINARY_MEDIA_PREFIX):
        return None
    payload = media_id[len(_BINARY_MEDIA_PREFIX) :]
    scene_seq_text, separator, encoded_raw_media_id = payload.partition(":")
    if separator == "" or encoded_raw_media_id == "":
        return None
    try:
        scene_seq = int(scene_seq_text)
    except ValueError:
        return None
    raw_media_id = unquote(encoded_raw_media_id).strip()
    if raw_media_id == "":
        return None
    return scene_seq, raw_media_id


def _encode_low_latency_frame(*, content: bytes, mime_type: str | None) -> tuple[bytes, str | None]:
    normalized_mime_type = str(mime_type or "").strip().lower()
    if normalized_mime_type == "image/png":
        return content, "image/png"
    if normalized_mime_type in {"image/jpeg", "image/jpg"}:
        return content, "image/jpeg"
    if Image is None:
        return content, mime_type
    try:
        image = Image.open(io.BytesIO(content))
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        return buffer.getvalue(), "image/jpeg"
    except Exception:
        return content, mime_type


def _decode_data_url(url: str) -> tuple[bytes, str | None]:
    header, separator, data = url.partition(",")
    if separator == "":
        raise ValueError("invalid_media_data_url")
    mime_type = header[5:].split(";", 1)[0].strip() or None
    if ";base64" in header:
        try:
            return base64.b64decode(data), mime_type
        except (ValueError, binascii.Error) as exc:
            raise ValueError("invalid_media_data_url") from exc
    return unquote_to_bytes(data), mime_type


def _load_media_content_from_ref(ref: MediaSourceRef) -> tuple[bytes, str | None] | None:
    url = str(ref.url or "").strip()
    if not url.startswith("data:"):
        return None
    return _decode_data_url(url)


def _load_rgb_frame_payload(payload: Mapping[str, object]) -> tuple[bytes, str | None] | None:
    for key in ("_rgb", "rgb", "rgb_array", "frame_rgb"):
        frame = payload.get(key)
        if frame is None:
            continue
        return _encode_rgb_frame(frame)
    return None


def _encode_rgb_frame(frame: object) -> tuple[bytes, str | None] | None:
    if Image is None:
        return None
    try:
        image = Image.fromarray(frame)  # type: ignore[arg-type]
    except Exception:
        return None
    try:
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        return buffer.getvalue(), "image/jpeg"
    except Exception:
        return None


__all__ = [
    "ArenaVisualFinishGate",
    "ArenaVisualLiveRegistry",
    "ArenaVisualLiveSessionSource",
    "RecorderLiveSessionSource",
]
