from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from dataclasses import dataclass, replace
from threading import RLock
from typing import Protocol
from urllib.parse import unquote_to_bytes

from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.role.arena.visualization.assembly import (
    assemble_visual_scene,
    collect_scene_media_refs,
)
from gage_eval.role.arena.visualization.contracts import (
    MediaSourceRef,
    ObserverRef,
    TimelineEvent,
    VisualScene,
    VisualSession,
)
from gage_eval.role.arena.visualization.gateway_service import TimelinePage
from gage_eval.role.arena.visualization.recorder import (
    ArenaVisualSessionRecorder,
)

_SUPPORTED_LIVE_SCENE_SCHEMES = {"http_pull"}


class ArenaVisualLiveSessionSource(Protocol):
    session_id: str
    run_id: str | None

    def load_session(self, *, observer: ObserverRef | None = None) -> VisualSession: ...

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


@dataclass(frozen=True, slots=True)
class RecorderLiveSessionSource:
    recorder: ArenaVisualSessionRecorder
    run_id: str | None = None
    visualization_spec: GameVisualizationSpec | None = None
    live_scene_scheme: str = "http_pull"

    def __post_init__(self) -> None:
        if self.live_scene_scheme not in _SUPPORTED_LIVE_SCENE_SCHEMES:
            raise ValueError(
                f"live_scene_scheme must be one of {sorted(_SUPPORTED_LIVE_SCENE_SCHEMES)!r}"
            )

    @property
    def session_id(self) -> str:
        return str(self.recorder.session_id)

    def load_session(self, *, observer: ObserverRef | None = None) -> VisualSession:
        state = self.recorder.export_live_state()
        return _override_observer(state.visual_session, observer=observer)

    def page_timeline(self, *, after_seq: int | None = None, limit: int = 50) -> TimelinePage:
        if limit <= 0:
            raise ValueError("limit must be positive")

        state = self.recorder.export_live_state()
        events = state.timeline_events
        start_index = 0
        if after_seq is not None:
            start_index = len(events)
            for index, event in enumerate(events):
                if event.seq > after_seq:
                    start_index = index
                    break
        page_events = events[start_index : start_index + limit]
        has_more = start_index + len(page_events) < len(events)
        next_after_seq = after_seq if not page_events else page_events[-1].seq
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
        state = self.recorder.export_live_state()
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

    def lookup_marker(self, marker: str) -> tuple[int, ...]:
        state = self.recorder.export_live_state()
        return tuple(state.marker_index.get(marker, ()))

    def lookup_media(self, media_id: str) -> MediaSourceRef | None:
        state = self.recorder.export_live_state()
        for event in state.timeline_events:
            scene = self.load_scene(seq=event.seq)
            if scene is None:
                continue
            refs = collect_scene_media_refs(scene)
            ref = refs.get(media_id)
            if ref is not None:
                return ref
        return None

    def load_media_content(self, media_id: str) -> tuple[bytes, str | None] | None:
        ref = self.lookup_media(media_id)
        if ref is None:
            return None
        url = str(ref.url or "").strip()
        if not url.startswith("data:"):
            return None
        return _decode_data_url(url)


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


__all__ = [
    "ArenaVisualLiveRegistry",
    "ArenaVisualLiveSessionSource",
    "RecorderLiveSessionSource",
]
