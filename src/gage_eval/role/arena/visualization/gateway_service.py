from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.role.arena.visualization.assembly import (
    assemble_visual_scene,
    collect_scene_media_refs,
)
from gage_eval.role.arena.visualization.contracts import (
    MediaSourceRef,
    ObserverRef,
    SeekSnapshotRecord,
    TimelineEvent,
    VisualScene,
    VisualSession,
)


@dataclass(frozen=True, slots=True)
class TimelinePage:
    events: tuple[TimelineEvent, ...]
    after_seq: int | None
    next_after_seq: int | None
    limit: int
    has_more: bool


@dataclass(frozen=True, slots=True)
class _LoadedVisualSession:
    manifest_path: Path
    manifest_payload: dict[str, Any]
    index_payload: dict[str, Any]
    visual_session: VisualSession
    timeline_events: tuple[TimelineEvent, ...]
    event_index: dict[int, TimelineEvent]
    snapshot_anchors: tuple[dict[str, Any], ...]
    seek_snapshots: tuple[SeekSnapshotRecord, ...]


class ArenaVisualGatewayQueryService:
    def __init__(self, *, visualization_spec: GameVisualizationSpec | None = None) -> None:
        self._visualization_spec = visualization_spec
        self._session_cache: dict[Path, _LoadedVisualSession] = {}
        self._scene_cache: dict[tuple[Path, int, str, str], VisualScene] = {}
        self._media_cache: dict[Path, dict[str, MediaSourceRef]] = {}

    def load_session(
        self,
        manifest_path: str | Path,
        *,
        observer: ObserverRef | None = None,
    ) -> VisualSession:
        bundle = self._load_bundle(manifest_path)
        return self._session_for_observer(bundle.visual_session, observer=observer)

    def page_timeline(
        self,
        manifest_path: str | Path,
        *,
        after_seq: int | None = None,
        limit: int = 50,
    ) -> TimelinePage:
        if limit <= 0:
            raise ValueError("limit must be positive")

        bundle = self._load_bundle(manifest_path)
        start_index = 0
        if after_seq is not None:
            start_index = len(bundle.timeline_events)
            for index, event in enumerate(bundle.timeline_events):
                if event.seq > after_seq:
                    start_index = index
                    break

        events = bundle.timeline_events[start_index : start_index + limit]
        has_more = start_index + len(events) < len(bundle.timeline_events)
        next_after_seq = after_seq if not events else events[-1].seq
        return TimelinePage(
            events=events,
            after_seq=after_seq,
            next_after_seq=next_after_seq,
            limit=limit,
            has_more=has_more,
        )

    def load_scene(
        self,
        manifest_path: str | Path,
        *,
        seq: int,
        observer: ObserverRef | None = None,
    ) -> VisualScene | None:
        bundle = self._load_bundle(manifest_path)
        visual_session = self._session_for_observer(bundle.visual_session, observer=observer)
        cache_key = (
            bundle.manifest_path,
            int(seq),
            visual_session.observer.observer_kind,
            visual_session.observer.observer_id,
        )
        cached = self._scene_cache.get(cache_key)
        if cached is not None:
            return cached

        event = bundle.event_index.get(int(seq))
        if event is None:
            return None

        snapshot_anchor = self._select_seek_snapshot(bundle.seek_snapshots, seq=event.seq, event=event)
        if snapshot_anchor is not None:
            snapshot_anchor = self._merge_snapshot_anchor_metadata(
                snapshot_anchor,
                snapshot_anchors=bundle.snapshot_anchors,
            )
        if snapshot_anchor is None:
            snapshot_anchor = self._select_snapshot_anchor(bundle.snapshot_anchors, seq=event.seq, event=event)
        snapshot_body = self._load_snapshot_body(bundle.manifest_path, snapshot_anchor)
        scene = assemble_visual_scene(
            visual_session=visual_session,
            event=event,
            snapshot_anchor=snapshot_anchor,
            snapshot_body=snapshot_body,
            visualization_spec=self._visualization_spec,
        )
        self._scene_cache[cache_key] = scene
        self._cache_media_refs(bundle.manifest_path, scene)
        return scene

    def lookup_marker(self, manifest_path: str | Path, marker: str) -> tuple[int, ...]:
        bundle = self._load_bundle(manifest_path)
        markers = bundle.index_payload.get("markers") or bundle.manifest_payload.get("timeline", {}).get("markers") or {}
        marker_payload = markers.get(marker)
        if isinstance(marker_payload, Sequence) and not isinstance(marker_payload, (str, bytes)):
            return tuple(int(item) for item in marker_payload)
        if marker_payload:
            return tuple(event.seq for event in bundle.timeline_events if event.type == marker)
        return ()

    def lookup_media(self, manifest_path: str | Path, media_id: str) -> MediaSourceRef | None:
        bundle = self._load_bundle(manifest_path)
        cached = self._media_cache.get(bundle.manifest_path, {}).get(media_id)
        if cached is not None:
            return cached

        for anchor in bundle.snapshot_anchors:
            scene = self.load_scene(bundle.manifest_path, seq=int(anchor["seq"]))
            if scene is not None:
                cached = self._media_cache.get(bundle.manifest_path, {}).get(media_id)
                if cached is not None:
                    return cached

        for event in bundle.timeline_events:
            scene = self.load_scene(bundle.manifest_path, seq=event.seq)
            if scene is not None:
                cached = self._media_cache.get(bundle.manifest_path, {}).get(media_id)
                if cached is not None:
                    return cached
        return None

    def _load_bundle(self, manifest_path: str | Path) -> _LoadedVisualSession:
        normalized_manifest_path = Path(manifest_path).expanduser().resolve()
        cached = self._session_cache.get(normalized_manifest_path)
        if cached is not None:
            return cached

        manifest_payload = json.loads(normalized_manifest_path.read_text(encoding="utf-8"))
        visual_session = VisualSession.from_dict(manifest_payload["visualSession"])
        index_payload = self._load_index_payload(normalized_manifest_path, manifest_payload)
        timeline_events = self._load_timeline_events(normalized_manifest_path, manifest_payload)
        snapshot_anchors = tuple(
            self._normalize_snapshot_anchor(normalized_manifest_path, anchor)
            for anchor in self._load_snapshot_anchors(manifest_payload, index_payload)
        )
        seek_snapshots = self._load_seek_snapshots(
            normalized_manifest_path,
            manifest_payload,
            snapshot_anchors,
        )
        bundle = _LoadedVisualSession(
            manifest_path=normalized_manifest_path,
            manifest_payload=manifest_payload,
            index_payload=index_payload,
            visual_session=visual_session,
            timeline_events=timeline_events,
            event_index={event.seq: event for event in timeline_events},
            snapshot_anchors=snapshot_anchors,
            seek_snapshots=seek_snapshots,
        )
        self._session_cache[normalized_manifest_path] = bundle
        return bundle

    def _session_for_observer(
        self,
        visual_session: VisualSession,
        *,
        observer: ObserverRef | None,
    ) -> VisualSession:
        if observer is None:
            return visual_session
        return replace(visual_session, observer=observer)

    def _load_index_payload(
        self,
        manifest_path: Path,
        manifest_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        ref = self._require_ref(
            manifest_payload=manifest_payload,
            artifacts_key="index_ref",
            timeline_key="indexRef",
            label="indexRef",
        )
        index_path = self._resolve_ref_path(manifest_path, ref)
        self._require_existing_file(index_path, label="indexRef")
        return dict(json.loads(index_path.read_text(encoding="utf-8")))

    def _load_timeline_events(
        self,
        manifest_path: Path,
        manifest_payload: Mapping[str, Any],
    ) -> tuple[TimelineEvent, ...]:
        ref = self._require_ref(
            manifest_payload=manifest_payload,
            artifacts_key="timeline_ref",
            timeline_key="timelineRef",
            label="timelineRef",
        )
        timeline_path = self._resolve_ref_path(manifest_path, ref)
        self._require_existing_file(timeline_path, label="timelineRef")

        events: list[TimelineEvent] = []
        for line in timeline_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(TimelineEvent.from_dict(json.loads(line)))
        return tuple(events)

    def _load_snapshot_anchors(
        self,
        manifest_payload: Mapping[str, Any],
        index_payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        anchors = index_payload.get("snapshotAnchors")
        if isinstance(anchors, Sequence) and not isinstance(anchors, (str, bytes)):
            return tuple(dict(anchor) for anchor in anchors if isinstance(anchor, Mapping))
        anchors = manifest_payload.get("timeline", {}).get("snapshotAnchors", ())
        if isinstance(anchors, Sequence) and not isinstance(anchors, (str, bytes)):
            return tuple(dict(anchor) for anchor in anchors if isinstance(anchor, Mapping))
        return ()

    def _normalize_snapshot_anchor(self, manifest_path: Path, anchor: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(anchor)
        if "snapshotRef" not in normalized or not normalized.get("snapshotRef"):
            raise ValueError("snapshotRef is required for snapshot anchors")
        snapshot_path = self._resolve_ref_path(manifest_path, normalized["snapshotRef"])
        self._require_existing_file(snapshot_path, label="snapshotRef")
        normalized["snapshotRef"] = str(snapshot_path)
        return normalized

    def _load_seek_snapshots(
        self,
        manifest_path: Path,
        manifest_payload: Mapping[str, Any],
        snapshot_anchors: Sequence[Mapping[str, Any]],
    ) -> tuple[SeekSnapshotRecord, ...]:
        ref = manifest_payload.get("artifacts", {}).get("seek_snapshots_ref")
        if ref:
            seek_snapshots_path = self._resolve_ref_path(manifest_path, str(ref))
            if seek_snapshots_path.exists():
                payload = json.loads(seek_snapshots_path.read_text(encoding="utf-8"))
                records = payload.get("seekSnapshots", ())
                if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
                    return tuple(
                        self._normalize_seek_snapshot_record(manifest_path, record)
                        for record in records
                        if isinstance(record, Mapping)
                    )
                return ()

        return tuple(
            SeekSnapshotRecord(
                seq=int(anchor["seq"]),
                ts_ms=int(anchor.get("tsMs", 0)),
                snapshot_mode="full",
                snapshot_ref=str(anchor["snapshotRef"]),
            )
            for anchor in snapshot_anchors
        )

    def _normalize_seek_snapshot_record(
        self,
        manifest_path: Path,
        payload: Mapping[str, Any],
    ) -> SeekSnapshotRecord:
        record = SeekSnapshotRecord.from_dict(payload)
        snapshot_path = self._resolve_ref_path(manifest_path, record.snapshot_ref)
        self._require_existing_file(snapshot_path, label="snapshotRef")
        return SeekSnapshotRecord(
            seq=record.seq,
            ts_ms=record.ts_ms,
            snapshot_mode=record.snapshot_mode,
            snapshot_ref=str(snapshot_path),
        )

    def _select_snapshot_anchor(
        self,
        snapshot_anchors: Sequence[Mapping[str, Any]],
        *,
        seq: int,
        event: TimelineEvent,
    ) -> dict[str, Any] | None:
        requested_snapshot_seq = event.ref_snapshot_seq
        if requested_snapshot_seq is not None:
            for anchor in snapshot_anchors:
                if int(anchor.get("seq", -1)) == int(requested_snapshot_seq):
                    return dict(anchor)

        best: Mapping[str, Any] | None = None
        for anchor in snapshot_anchors:
            anchor_seq = int(anchor.get("seq", -1))
            if anchor_seq <= seq and (best is None or anchor_seq > int(best.get("seq", -1))):
                best = anchor
        return None if best is None else dict(best)

    def _select_seek_snapshot(
        self,
        seek_snapshots: Sequence[SeekSnapshotRecord],
        *,
        seq: int,
        event: TimelineEvent,
    ) -> dict[str, Any] | None:
        requested_snapshot_seq = event.ref_snapshot_seq
        if requested_snapshot_seq is not None:
            for record in seek_snapshots:
                if record.seq == int(requested_snapshot_seq):
                    return record.to_dict()
            return None

        best: SeekSnapshotRecord | None = None
        for record in seek_snapshots:
            if record.seq <= seq and (best is None or record.seq > best.seq):
                best = record
        return None if best is None else best.to_dict()

    def _merge_snapshot_anchor_metadata(
        self,
        snapshot_anchor: Mapping[str, Any],
        *,
        snapshot_anchors: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        merged = dict(snapshot_anchor)
        snapshot_seq = int(merged.get("seq", -1))
        for legacy_anchor in snapshot_anchors:
            if int(legacy_anchor.get("seq", -1)) != snapshot_seq:
                continue
            for key, value in legacy_anchor.items():
                if key == "snapshotRef":
                    continue
                merged.setdefault(key, value)
            break
        return merged

    def _load_snapshot_body(
        self,
        manifest_path: Path,
        snapshot_anchor: Mapping[str, Any] | None,
    ) -> Any:
        if snapshot_anchor is None:
            return None
        snapshot_ref = snapshot_anchor.get("snapshotRef")
        if not snapshot_ref:
            raise ValueError("snapshotRef is required for snapshot anchors")
        snapshot_path = self._resolve_ref_path(manifest_path, snapshot_ref)
        self._require_existing_file(snapshot_path, label="snapshotRef")
        snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if isinstance(snapshot_payload, Mapping):
            return snapshot_payload.get("body")
        return None

    def _resolve_ref_path(self, manifest_path: Path, ref: str | Path) -> Path:
        ref_path = Path(ref).expanduser()
        if ref_path.is_absolute():
            return ref_path.resolve()
        return (manifest_path.parent / ref_path).resolve()

    def _require_ref(
        self,
        *,
        manifest_payload: Mapping[str, Any],
        artifacts_key: str,
        timeline_key: str,
        label: str,
    ) -> str:
        ref = (
            manifest_payload.get("artifacts", {}).get(artifacts_key)
            or manifest_payload.get("timeline", {}).get(timeline_key)
        )
        if not ref:
            raise ValueError(f"{label} is required for arena_visual_session bundles")
        return str(ref)

    def _require_existing_file(self, path: Path, *, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{label} target does not exist: {path}")

    def _cache_media_refs(self, manifest_path: Path, scene: VisualScene) -> None:
        media_cache = self._media_cache.setdefault(manifest_path, {})
        media_cache.update(collect_scene_media_refs(scene))
