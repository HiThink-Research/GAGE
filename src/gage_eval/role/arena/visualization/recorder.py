from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gage_eval.role.arena.types import GameResult
from gage_eval.role.arena.visualization.artifacts import (
    ArenaVisualArtifactLayout,
    ArenaVisualSessionArtifacts,
    build_visual_session_manifest,
    to_bounded_json_safe,
    to_visual_json_safe,
)
from gage_eval.role.arena.visualization.contracts import (
    ObserverRef,
    PlaybackState,
    SchedulingState,
    SeekSnapshotRecord,
    TimelineEvent,
    VisualSession,
)


@dataclass(slots=True)
class ArenaVisualSessionRecorder:
    plugin_id: str
    game_id: str
    scheduling_family: str
    session_id: str
    lifecycle: str = "initializing"
    playback_mode: str = "live_tail"
    observer_id: str = ""
    observer_kind: str = "spectator"
    _seq: int = field(default=0, init=False, repr=False)
    _events: list[TimelineEvent] = field(default_factory=list, init=False, repr=False)
    _snapshot_payloads: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _marker_index: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list), init=False, repr=False)
    _artifacts: ArenaVisualSessionArtifacts | None = field(default=None, init=False, repr=False)
    _latest_result: Any = field(default=None, init=False, repr=False)
    _scheduling_phase: str = field(default="idle", init=False, repr=False)
    _scheduling_accepts_human_intent: bool = field(default=False, init=False, repr=False)
    _scheduling_active_actor_id: str | None = field(default=None, init=False, repr=False)
    _scheduling_window_id: str | None = field(default=None, init=False, repr=False)

    @property
    def artifacts(self) -> ArenaVisualSessionArtifacts | None:
        return self._artifacts

    @property
    def visual_session_ref(self) -> str | None:
        if self._artifacts is None:
            return None
        return self._artifacts.visual_session_ref

    def record_decision_window_open(
        self,
        *,
        ts_ms: int | None = None,
        step: int,
        tick: int,
        player_id: str,
        observation: Any = None,
        window_id: str | None = None,
    ) -> TimelineEvent:
        self._scheduling_phase = "waiting_for_intent"
        self._scheduling_accepts_human_intent = True
        self._scheduling_active_actor_id = str(player_id)
        self._scheduling_window_id = window_id
        return self._record_event(
            "decision_window_open",
            label="decision_window_open",
            ts_ms=ts_ms,
            payload={
                "step": int(step),
                "tick": int(tick),
                "playerId": str(player_id),
                "windowId": window_id,
                "observation": to_bounded_json_safe(observation),
            },
        )

    def record_action_intent(
        self,
        *,
        ts_ms: int | None = None,
        step: int,
        tick: int,
        player_id: str,
        action: Any,
        observation: Any = None,
        intent_id: str | None = None,
    ) -> TimelineEvent:
        self._scheduling_phase = "waiting_for_intent"
        self._scheduling_active_actor_id = str(player_id)
        return self._record_event(
            "action_intent",
            label="action_intent",
            ts_ms=ts_ms,
            payload={
                "step": int(step),
                "tick": int(tick),
                "playerId": str(player_id),
                "intentId": intent_id,
                "action": to_bounded_json_safe(action),
                "observation": to_bounded_json_safe(observation),
            },
        )

    def record_action_committed(
        self,
        *,
        ts_ms: int | None = None,
        step: int,
        tick: int,
        player_id: str,
        action: Any,
        trace_entry: Any = None,
        result: Any = None,
    ) -> TimelineEvent:
        self._scheduling_phase = "advancing"
        self._scheduling_accepts_human_intent = False
        self._scheduling_active_actor_id = str(player_id)
        return self._record_event(
            "action_committed",
            label="action_committed",
            ts_ms=ts_ms,
            payload={
                "step": int(step),
                "tick": int(tick),
                "playerId": str(player_id),
                "action": to_bounded_json_safe(action),
                "traceEntry": to_bounded_json_safe(trace_entry),
                "result": to_bounded_json_safe(result),
            },
        )

    def record_decision_window_close(
        self,
        *,
        ts_ms: int | None = None,
        step: int,
        tick: int,
        player_id: str,
        window_id: str | None = None,
        reason: str | None = None,
    ) -> TimelineEvent:
        self._scheduling_phase = "advancing"
        self._scheduling_accepts_human_intent = False
        self._scheduling_active_actor_id = str(player_id)
        self._scheduling_window_id = window_id or self._scheduling_window_id
        return self._record_event(
            "decision_window_close",
            label="decision_window_close",
            ts_ms=ts_ms,
            payload={
                "step": int(step),
                "tick": int(tick),
                "playerId": str(player_id),
                "windowId": window_id or self._scheduling_window_id,
                "reason": reason,
            },
        )

    def record_snapshot(
        self,
        *,
        ts_ms: int | None = None,
        step: int,
        tick: int,
        snapshot: Any,
        label: str = "snapshot",
        anchor: bool = True,
    ) -> TimelineEvent:
        self._scheduling_phase = "recording"
        event = self._record_event(
            "snapshot",
            label=label,
            ts_ms=ts_ms,
            payload={
                "step": int(step),
                "tick": int(tick),
                "anchor": bool(anchor),
                "snapshot": to_bounded_json_safe(snapshot),
            },
        )
        if anchor:
            self._snapshot_payloads.append(
                {
                    "seq": event.seq,
                    "tsMs": event.ts_ms,
                    "label": label,
                    "snapshot": to_visual_json_safe(snapshot),
                }
            )
        return event

    def record_result(
        self,
        *,
        ts_ms: int | None = None,
        step: int,
        tick: int,
        result: Any,
    ) -> TimelineEvent:
        self._latest_result = result
        self._scheduling_phase = "completed"
        self.lifecycle = "live_ended"
        return self._record_event(
            "result",
            label="result",
            ts_ms=ts_ms,
            lifecycle="live_ended",
            payload={
                "step": int(step),
                "tick": int(tick),
                "result": to_bounded_json_safe(result),
            },
        )

    def build_visual_session(self) -> VisualSession:
        return VisualSession(
            session_id=self.session_id,
            game_id=self.game_id,
            plugin_id=self.plugin_id,
            lifecycle=self.lifecycle,
            playback=PlaybackState(mode=self.playback_mode),
            observer=ObserverRef(
                observer_id=self.observer_id,
                observer_kind=self.observer_kind,
            ),
            scheduling=SchedulingState(
                family=self.scheduling_family,
                phase=self._scheduling_phase,
                accepts_human_intent=self._scheduling_accepts_human_intent,
                active_actor_id=self._scheduling_active_actor_id,
                window_id=self._scheduling_window_id,
            ),
            capabilities={
                "supportsReplay": True,
                "supportsTimeline": True,
            },
            summary=self._build_summary(),
            timeline=self._build_timeline_manifest(),
        )

    def persist(self, replay_path: str | Path) -> ArenaVisualSessionArtifacts:
        layout = ArenaVisualArtifactLayout.from_replay_path(replay_path)
        if self._artifacts is not None and self._artifacts.layout.replay_path == layout.replay_path:
            return self._artifacts

        layout.session_dir.mkdir(parents=True, exist_ok=True)
        layout.snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot_anchors: list[dict[str, Any]] = []
        seek_snapshots: list[SeekSnapshotRecord] = []
        for snapshot in self._snapshot_payloads:
            snapshot_path = layout.snapshot_dir / f"seq-{int(snapshot['seq']):06d}.json"
            snapshot_payload = {
                "seq": int(snapshot["seq"]),
                "tsMs": int(snapshot["tsMs"]),
                "label": snapshot.get("label"),
                "anchor": True,
                "body": snapshot.get("snapshot"),
            }
            snapshot_path.write_text(
                json.dumps(snapshot_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            snapshot_anchors.append(
                {
                    "seq": int(snapshot["seq"]),
                    "tsMs": int(snapshot["tsMs"]),
                    "label": snapshot.get("label"),
                    "snapshotRef": str(snapshot_path),
                }
            )
            seek_snapshots.append(
                SeekSnapshotRecord(
                    seq=int(snapshot["seq"]),
                    ts_ms=int(snapshot["tsMs"]),
                    snapshot_mode="full",
                    snapshot_ref=str(snapshot_path),
                )
            )

        timeline_path = layout.timeline_path
        timeline_path.write_text(
            "".join(
                json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n"
                for event in self._events
            ),
            encoding="utf-8",
        )

        self.lifecycle = "closed"
        visual_session = self.build_visual_session()
        manifest_payload, index_payload = build_visual_session_manifest(
            layout=layout,
            visual_session=visual_session,
            timeline_events=tuple(self._events),
            snapshot_anchors=tuple(snapshot_anchors),
            marker_index={key: tuple(value) for key, value in self._marker_index.items()},
            result=self._latest_result,
        )
        layout.manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        layout.index_path.write_text(
            json.dumps(index_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        layout.seek_snapshots_path.write_text(
            json.dumps(
                {
                    "schema": "arena_visual_session/v1",
                    "version": "1.0.0",
                    "seekSnapshots": [
                        {
                            **record.to_dict(),
                            "snapshotRef": layout.relative_ref(Path(record.snapshot_ref)),
                        }
                        for record in seek_snapshots
                    ],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        persisted_session = VisualSession.from_dict(manifest_payload["visualSession"])
        self._artifacts = ArenaVisualSessionArtifacts(
            layout=layout,
            visual_session=persisted_session,
            timeline_events=tuple(self._events),
            snapshot_anchors=tuple(snapshot_anchors),
            marker_index={key: tuple(value) for key, value in self._marker_index.items()},
            manifest_payload=manifest_payload,
            index_payload=index_payload,
        )
        return self._artifacts

    def _record_event(
        self,
        event_type: str,
        *,
        label: str,
        ts_ms: int | None,
        lifecycle: str | None = "live_running",
        payload: Any,
    ) -> TimelineEvent:
        self._seq += 1
        event = TimelineEvent(
            seq=self._seq,
            ts_ms=int(ts_ms if ts_ms is not None else _now_ms()),
            type=event_type,
            label=label,
            payload=payload,
        )
        self._events.append(event)
        self._marker_index[event.type].append(event.seq)
        if lifecycle is not None:
            self.lifecycle = lifecycle
        return event

    def _build_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "eventCount": len(self._events),
            "snapshotCount": len(self._snapshot_payloads),
        }
        if self._latest_result is not None:
            summary["result"] = to_bounded_json_safe(self._latest_result)
        return summary

    def _build_timeline_manifest(self) -> dict[str, Any]:
        return {
            "eventCount": len(self._events),
            "headSeq": self._events[0].seq if self._events else 0,
            "tailSeq": self._events[-1].seq if self._events else 0,
            "markers": {key: len(value) for key, value in self._marker_index.items()},
            "snapshotAnchors": [
                {
                    "seq": int(snapshot["seq"]),
                    "tsMs": int(snapshot["tsMs"]),
                    "label": snapshot.get("label"),
                }
                for snapshot in self._snapshot_payloads
            ],
        }


def _now_ms() -> int:
    return int(time.time() * 1000)
