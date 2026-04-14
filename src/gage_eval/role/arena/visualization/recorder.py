from __future__ import annotations

import copy
import json
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from loguru import logger

from gage_eval.role.arena.types import GameResult
from gage_eval.role.arena.visualization.artifacts import (
    ArenaVisualArtifactLayout,
    ArenaVisualSessionArtifacts,
    build_visual_session_manifest,
    to_bounded_json_safe,
    to_scene_json_safe,
)
from gage_eval.role.arena.visualization.contracts import (
    ObserverRef,
    PlaybackState,
    SchedulingState,
    SeekSnapshotRecord,
    TimelineEvent,
    VisualSession,
    ControlCommand,
)


@dataclass(frozen=True, slots=True)
class ArenaVisualLiveState:
    visual_session: VisualSession
    timeline_events: tuple[TimelineEvent, ...]
    snapshot_payloads: tuple[dict[str, Any], ...]
    marker_index: dict[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class PendingSnapshotRecord:
    ts_ms: int | None
    step: int
    tick: int
    snapshot: Any
    label: str = "snapshot"
    anchor: bool = True
    snapshot_is_scene_safe: bool = False


@dataclass(slots=True)
class ArenaVisualSessionRecorder:
    plugin_id: str
    game_id: str
    scheduling_family: str
    session_id: str
    observer_modes: tuple[str, ...] = ()
    visual_kind: str | None = None
    extra_capabilities: dict[str, Any] = field(default_factory=dict)
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
    _playback_cursor_ts: int = field(default=0, init=False, repr=False)
    _playback_cursor_event_seq: int = field(default=0, init=False, repr=False)
    _playback_speed: float = field(default=1.0, init=False, repr=False)
    _runtime_metrics: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _pending_snapshots: list[PendingSnapshotRecord] = field(default_factory=list, init=False, repr=False)
    _max_pending_snapshots: int = field(default=128, init=False, repr=False)
    _live_revision: int = field(default=0, init=False, repr=False)
    _background_snapshot_poll_interval_s: float = field(default=0.01, init=False, repr=False)
    _snapshot_drain_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _snapshot_drain_stop: threading.Event | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _revision_condition: threading.Condition = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._revision_condition = threading.Condition(self._lock)

    @property
    def artifacts(self) -> ArenaVisualSessionArtifacts | None:
        with self._lock:
            return self._artifacts

    @property
    def visual_session_ref(self) -> str | None:
        with self._lock:
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
        accepts_human_intent: bool = True,
    ) -> TimelineEvent:
        with self._lock:
            self.set_scheduling_state(
                phase="waiting_for_intent",
                accepts_human_intent=accepts_human_intent,
                active_actor_id=player_id,
                window_id=window_id,
            )
            return self._record_event(
                "decision_window_open",
                label="decision_window_open",
                ts_ms=ts_ms,
                payload={
                    "step": int(step),
                    "tick": int(tick),
                    "playerId": str(player_id),
                    "windowId": window_id,
                    "observation": to_scene_json_safe(observation),
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
        with self._lock:
            self.set_scheduling_state(
                phase="waiting_for_intent",
                accepts_human_intent=self._scheduling_accepts_human_intent,
                active_actor_id=player_id,
                window_id=self._scheduling_window_id,
            )
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
                    "observation": to_scene_json_safe(observation),
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
        with self._lock:
            self.set_scheduling_state(
                phase="advancing",
                accepts_human_intent=False,
                active_actor_id=player_id,
                window_id=self._scheduling_window_id,
            )
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
                    "result": to_scene_json_safe(result),
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
        with self._lock:
            self.set_scheduling_state(
                phase="advancing",
                accepts_human_intent=False,
                active_actor_id=player_id,
                window_id=window_id or self._scheduling_window_id,
            )
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
        snapshot_is_scene_safe: bool = False,
    ) -> TimelineEvent:
        with self._lock:
            return self._record_snapshot_locked(
                ts_ms=ts_ms,
                step=step,
                tick=tick,
                snapshot=snapshot,
                label=label,
                anchor=anchor,
                snapshot_is_scene_safe=snapshot_is_scene_safe,
            )

    def enqueue_snapshot(
        self,
        *,
        ts_ms: int | None = None,
        step: int,
        tick: int,
        snapshot: Any,
        label: str = "snapshot",
        anchor: bool = True,
        snapshot_is_scene_safe: bool = False,
    ) -> None:
        with self._lock:
            max_pending = max(1, int(self._max_pending_snapshots))
            if len(self._pending_snapshots) >= max_pending:
                self._pending_snapshots.pop(0)
                dropped = int(self._runtime_metrics.get("dropped_snapshot_count") or 0)
                next_dropped = dropped + 1
                self._runtime_metrics["dropped_snapshot_count"] = next_dropped
                if dropped == 0 or next_dropped % 100 == 0:
                    logger.warning(
                        "ArenaVisualSessionRecorder dropped pending snapshot, total={}",
                        next_dropped,
                    )
            self._pending_snapshots.append(
                PendingSnapshotRecord(
                    ts_ms=ts_ms,
                    step=int(step),
                    tick=int(tick),
                    snapshot=snapshot,
                    label=str(label),
                    anchor=bool(anchor),
                    snapshot_is_scene_safe=bool(snapshot_is_scene_safe),
                )
            )
            self._revision_condition.notify_all()

    def pending_snapshot_count(self) -> int:
        with self._lock:
            return len(self._pending_snapshots)

    def flush_pending_snapshots(self) -> int:
        with self._lock:
            return self._drain_pending_snapshots_locked()

    def start_background_snapshot_drain(self, *, poll_interval_s: float = 0.01) -> None:
        normalized_interval_s = max(0.001, float(poll_interval_s))
        with self._lock:
            thread = self._snapshot_drain_thread
            if thread is not None and thread.is_alive():
                self._background_snapshot_poll_interval_s = normalized_interval_s
                return
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._background_snapshot_drain_worker,
                name=f"arena-visual-snapshot-drain:{self.session_id}",
                daemon=True,
            )
            self._background_snapshot_poll_interval_s = normalized_interval_s
            self._snapshot_drain_stop = stop_event
            self._snapshot_drain_thread = thread
        thread.start()

    def stop_background_snapshot_drain(
        self,
        *,
        flush: bool = True,
        join_timeout_s: float = 1.0,
    ) -> None:
        with self._lock:
            thread = self._snapshot_drain_thread
            stop_event = self._snapshot_drain_stop
            self._snapshot_drain_thread = None
            self._snapshot_drain_stop = None
            self._revision_condition.notify_all()
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(join_timeout_s)))
        if flush:
            self.flush_pending_snapshots()

    def record_result(
        self,
        *,
        ts_ms: int | None = None,
        step: int,
        tick: int,
        result: Any,
    ) -> TimelineEvent:
        with self._lock:
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
                    "result": to_scene_json_safe(result),
                },
            )

    def build_visual_session(self) -> VisualSession:
        with self._lock:
            return self._build_visual_session_locked(include_snapshot_anchors=True)

    def build_live_session(self) -> VisualSession:
        with self._lock:
            self._drain_pending_snapshots_for_live_read_locked()
            return self._build_visual_session_locked(include_snapshot_anchors=False)

    def current_live_revision(self) -> int:
        with self._lock:
            return self._live_revision

    def wait_for_live_revision(self, *, after_revision: int, timeout_s: float | None = None) -> int:
        with self._lock:
            if self._live_revision > int(after_revision):
                return self._live_revision
            self._revision_condition.wait(
                timeout=None if timeout_s is None else max(0.0, float(timeout_s))
            )
            return self._live_revision

    def set_scheduling_state(
        self,
        *,
        phase: str,
        accepts_human_intent: bool,
        active_actor_id: str | None,
        window_id: str | None = None,
    ) -> None:
        with self._lock:
            next_phase = str(phase)
            next_accepts_human_intent = bool(accepts_human_intent)
            next_active_actor_id = None if active_actor_id is None else str(active_actor_id)
            next_window_id = window_id
            if (
                self._scheduling_phase == next_phase
                and self._scheduling_accepts_human_intent == next_accepts_human_intent
                and self._scheduling_active_actor_id == next_active_actor_id
                and self._scheduling_window_id == next_window_id
            ):
                return
            self._scheduling_phase = next_phase
            self._scheduling_accepts_human_intent = next_accepts_human_intent
            self._scheduling_active_actor_id = next_active_actor_id
            self._scheduling_window_id = next_window_id
            self._mark_live_revision_locked()

    def update_runtime_metrics(self, **metrics: Any) -> None:
        with self._lock:
            for key, value in metrics.items():
                if value is None:
                    self._runtime_metrics.pop(str(key), None)
                else:
                    self._runtime_metrics[str(key)] = value

    def reopen_live_round(self) -> int:
        with self._lock:
            self._drain_pending_snapshots_locked()
            self._latest_result = None
            self.lifecycle = "live_running"
            self.playback_mode = "live_tail"
            self._scheduling_phase = "idle"
            self._scheduling_accepts_human_intent = False
            self._scheduling_active_actor_id = None
            self._scheduling_window_id = None
            next_seq = self._tail_seq_locked()
            self._playback_cursor_event_seq = next_seq
            self._playback_cursor_ts = self._event_ts_for_seq_locked(next_seq)
            self._mark_live_revision_locked()
            return next_seq

    def export_live_state(self) -> ArenaVisualLiveState:
        with self._lock:
            return ArenaVisualLiveState(
                visual_session=self._build_visual_session_locked(include_snapshot_anchors=True),
                timeline_events=tuple(self._events),
                snapshot_payloads=tuple(copy.deepcopy(self._snapshot_payloads)),
                marker_index={key: tuple(value) for key, value in self._marker_index.items()},
            )

    def export_live_header(self) -> dict[str, Any]:
        with self._lock:
            self._drain_pending_snapshots_for_live_read_locked()
            cursor_event_seq = self._resolve_playback_cursor_event_seq_locked()
            return {
                "lifecycle": self.lifecycle,
                "cursorEventSeq": cursor_event_seq,
                "tailSeq": self._tail_seq_locked(),
            }

    def page_live_timeline(
        self,
        *,
        after_seq: int | None = None,
        limit: int = 50,
    ) -> tuple[tuple[TimelineEvent, ...], int | None, bool]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._lock:
            self._drain_pending_snapshots_for_live_read_locked()
            start_index = 0
            if after_seq is not None:
                start_index = len(self._events)
                for index, event in enumerate(self._events):
                    if event.seq > after_seq:
                        start_index = index
                        break
            events = tuple(self._events[start_index : start_index + limit])
            has_more = start_index + len(events) < len(self._events)
            next_after_seq = after_seq if not events else events[-1].seq
            return events, next_after_seq, has_more

    def lookup_marker(self, marker: str) -> tuple[int, ...]:
        with self._lock:
            self._drain_pending_snapshots_for_live_read_locked()
            return tuple(self._marker_index.get(str(marker), ()))

    def apply_control_command(self, command: ControlCommand) -> int:
        with self._lock:
            current_seq = self._resolve_playback_cursor_event_seq_locked()
            next_mode = self.playback_mode
            next_seq = current_seq
            next_speed = self._playback_speed

            if command.command_type == "pause":
                next_mode = "paused"
            elif command.command_type == "replay":
                next_mode = "replay_playing"
                next_seq = self._head_playback_seq_locked()
            elif command.command_type in {"follow_tail", "back_to_tail"}:
                next_mode = "live_tail"
                next_seq = self._tail_seq_locked()
            elif command.command_type == "seek_seq":
                next_mode = "paused"
                next_seq = self._resolve_target_seq_locked(command.target_seq)
            elif command.command_type == "seek_end":
                next_mode = "paused"
                next_seq = self._tail_seq_locked()
            elif command.command_type == "step":
                next_mode = "paused"
                next_seq = self._step_cursor_seq_locked(current_seq, command.step_delta)
            elif command.command_type == "set_speed":
                next_speed = float(command.speed if command.speed is not None else next_speed)

            self.playback_mode = next_mode
            self._playback_speed = next_speed
            self._playback_cursor_event_seq = next_seq
            self._playback_cursor_ts = self._event_ts_for_seq_locked(next_seq)
            self._mark_live_revision_locked()
            return next_seq

    def persist(self, replay_path: str | Path) -> ArenaVisualSessionArtifacts:
        with self._lock:
            self._drain_pending_snapshots_locked()
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
                        snapshot_mode=self._resolve_seek_snapshot_mode(),
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

            visual_session = replace(self.build_visual_session(), lifecycle="closed")
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

    def _build_visual_session_locked(self, *, include_snapshot_anchors: bool) -> VisualSession:
        return VisualSession(
            session_id=self.session_id,
            game_id=self.game_id,
            plugin_id=self.plugin_id,
            lifecycle=self.lifecycle,
            playback=self._build_playback_state_locked(),
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
                "supportsSeek": True,
                "observerModes": list(self.observer_modes),
                **copy.deepcopy(self.extra_capabilities),
            },
            summary=self._build_summary(),
            runtime_metrics=copy.deepcopy(self._runtime_metrics),
            timeline=self._build_timeline_manifest(
                include_snapshot_anchors=include_snapshot_anchors,
            ),
        )

    def _record_snapshot_locked(
        self,
        *,
        ts_ms: int | None,
        step: int,
        tick: int,
        snapshot: Any,
        label: str,
        anchor: bool,
        snapshot_is_scene_safe: bool,
    ) -> TimelineEvent:
        self._scheduling_phase = "recording"
        scene_safe_snapshot = snapshot if snapshot_is_scene_safe else to_scene_json_safe(snapshot)
        event = self._record_event(
            "snapshot",
            label=label,
            ts_ms=ts_ms,
            payload={
                "step": int(step),
                "tick": int(tick),
                "anchor": bool(anchor),
                "snapshot": scene_safe_snapshot,
            },
        )
        if anchor:
            self._snapshot_payloads.append(
                {
                    "seq": event.seq,
                    "tsMs": event.ts_ms,
                    "label": label,
                    "snapshot": scene_safe_snapshot,
                }
            )
        return event

    def _drain_pending_snapshots_locked(self) -> int:
        if not self._pending_snapshots:
            return 0
        pending = tuple(self._pending_snapshots)
        self._pending_snapshots.clear()
        for snapshot in pending:
            self._record_snapshot_locked(
                ts_ms=snapshot.ts_ms,
                step=snapshot.step,
                tick=snapshot.tick,
                snapshot=snapshot.snapshot,
                label=snapshot.label,
                anchor=snapshot.anchor,
                snapshot_is_scene_safe=snapshot.snapshot_is_scene_safe,
            )
        return len(pending)

    def _drain_pending_snapshots_for_live_read_locked(self) -> int:
        if self._background_snapshot_drain_active_locked():
            return 0
        return self._drain_pending_snapshots_locked()

    def _mark_live_revision_locked(self) -> None:
        self._live_revision += 1
        self._revision_condition.notify_all()

    def _background_snapshot_drain_active_locked(self) -> bool:
        thread = self._snapshot_drain_thread
        stop_event = self._snapshot_drain_stop
        return bool(
            thread is not None
            and thread.is_alive()
            and stop_event is not None
            and not stop_event.is_set()
        )

    def _background_snapshot_drain_worker(self) -> None:
        while True:
            with self._lock:
                stop_event = self._snapshot_drain_stop
                while (
                    stop_event is not None
                    and not stop_event.is_set()
                    and not self._pending_snapshots
                ):
                    self._revision_condition.wait(
                        timeout=max(0.001, self._background_snapshot_poll_interval_s)
                    )
                    stop_event = self._snapshot_drain_stop
                if stop_event is None or stop_event.is_set():
                    self._drain_pending_snapshots_locked()
                    return
                self._drain_pending_snapshots_locked()

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
        self._mark_live_revision_locked()
        self._marker_index[event.type].append(event.seq)
        if lifecycle is not None:
            self.lifecycle = lifecycle
        if self.playback_mode == "live_tail":
            self._playback_cursor_event_seq = event.seq
            self._playback_cursor_ts = event.ts_ms
        return event

    def _build_playback_state_locked(self) -> PlaybackState:
        cursor_event_seq = self._resolve_playback_cursor_event_seq_locked()
        return PlaybackState(
            mode=self.playback_mode,
            cursor_ts=self._event_ts_for_seq_locked(cursor_event_seq),
            cursor_event_seq=cursor_event_seq,
            speed=self._playback_speed,
            can_seek=True,
        )

    def _resolve_playback_cursor_event_seq_locked(self) -> int:
        if not self._events:
            return 0
        if self.playback_mode == "live_tail":
            return self._tail_seq_locked()
        if self._playback_cursor_event_seq <= 0:
            return self._tail_seq_locked()
        return self._resolve_target_seq_locked(self._playback_cursor_event_seq)

    def _resolve_target_seq_locked(self, target_seq: int | None) -> int:
        if not self._events:
            return 0
        if target_seq is None:
            return self._tail_seq_locked()
        selected = self._head_seq_locked()
        normalized_target = int(target_seq)
        for event in self._events:
            if event.seq > normalized_target:
                break
            selected = event.seq
        return selected

    def _step_cursor_seq_locked(self, current_seq: int, step_delta: int | None) -> int:
        if not self._events:
            return 0
        checkpoints = self._playback_checkpoint_seqs_locked()
        if not checkpoints:
            return self._resolve_target_seq_locked(current_seq)
        normalized_current = self._resolve_target_seq_locked(current_seq)
        normalized_delta = int(step_delta or 0)
        if normalized_delta > 0:
            for seq in checkpoints:
                if seq > normalized_current:
                    return seq
            return checkpoints[-1]
        if normalized_delta < 0:
            for seq in reversed(checkpoints):
                if seq < normalized_current:
                    return seq
            return checkpoints[0]
        return normalized_current

    def _head_playback_seq_locked(self) -> int:
        checkpoints = self._playback_checkpoint_seqs_locked()
        if checkpoints:
            return checkpoints[0]
        return self._head_seq_locked()

    def _playback_checkpoint_seqs_locked(self) -> list[int]:
        turn_based = [
            int(event.seq)
            for event in self._events
            if event.type in {"decision_window_open", "result"}
        ]
        if turn_based:
            return turn_based

        generic = [
            int(event.seq)
            for event in self._events
            if event.type in {"snapshot", "frame_ref", "result"}
        ]
        if generic:
            return generic
        return [int(event.seq) for event in self._events]

    def _head_seq_locked(self) -> int:
        return self._events[0].seq if self._events else 0

    def _tail_seq_locked(self) -> int:
        return self._events[-1].seq if self._events else 0

    def _event_ts_for_seq_locked(self, seq: int) -> int:
        normalized_seq = int(seq)
        if normalized_seq <= 0:
            return 0
        for event in self._events:
            if event.seq == normalized_seq:
                return event.ts_ms
        return 0

    def _build_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "eventCount": len(self._events),
            "snapshotCount": len(self._snapshot_payloads),
        }
        if self._runtime_metrics:
            summary["realtimeMetrics"] = copy.deepcopy(self._runtime_metrics)
        if self._latest_result is not None:
            summary["result"] = to_bounded_json_safe(self._latest_result)
        return summary

    def _resolve_seek_snapshot_mode(self) -> str:
        visual_kind = self._resolve_visual_kind()
        if visual_kind == "frame":
            return "media_ref"
        return "full"

    def _resolve_visual_kind(self) -> str | None:
        if self.visual_kind in {"board", "table", "frame", "rts"}:
            return str(self.visual_kind)
        if self.plugin_id.endswith(".board_v1"):
            return "board"
        if self.plugin_id.endswith(".table_v1"):
            return "table"
        if self.plugin_id.endswith(".frame_v1"):
            return "frame"
        return None

    def _build_timeline_manifest(self, *, include_snapshot_anchors: bool) -> dict[str, Any]:
        manifest = {
            "eventCount": len(self._events),
            "headSeq": self._events[0].seq if self._events else 0,
            "tailSeq": self._events[-1].seq if self._events else 0,
            "markers": {key: len(value) for key, value in self._marker_index.items()},
        }
        if include_snapshot_anchors:
            manifest["snapshotAnchors"] = [
                {
                    "seq": int(snapshot["seq"]),
                    "tsMs": int(snapshot["tsMs"]),
                    "label": snapshot.get("label"),
                }
                for snapshot in self._snapshot_payloads
            ]
        else:
            manifest["snapshotAnchors"] = []
        return manifest


def _now_ms() -> int:
    return int(time.time() * 1000)
