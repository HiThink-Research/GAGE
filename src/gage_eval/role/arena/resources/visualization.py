"""Compatibility wrapper for the legacy visualization session API.

New JSON-safe contracts live in :mod:`gage_eval.role.arena.visualization.contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from gage_eval.role.arena.visualization.contracts import (
    ActionIntentReceipt,
    MediaSourceRef,
    TimelineEvent,
    VisualScene,
    VisualSession,
)


class VisualizationPhase(str, Enum):
    LIVE = "live"
    PERSISTING = "persisting"
    REPLAY_READY = "replay_ready"
    RELEASED = "released"


@dataclass
class VisualizationSession:
    phase: VisualizationPhase
    display: object
    replay_viewer: object
    resource_category: str = "visualization_resource"
    phase_history: list[VisualizationPhase] = field(default_factory=list)
    artifacts: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.phase_history.append(self.phase)
        self._sync_artifacts()

    def switch_to_replay(self, replay_uri: str) -> None:
        self._set_phase(VisualizationPhase.PERSISTING)
        self.display.close_inputs()
        self.replay_viewer.load(replay_uri)
        self.artifacts["replay_ref"] = replay_uri
        self._set_phase(VisualizationPhase.REPLAY_READY)

    def release(self) -> None:
        closer = getattr(self.replay_viewer, "close", None)
        if callable(closer):
            closer()
        self._set_phase(VisualizationPhase.RELEASED)

    def _set_phase(self, phase: VisualizationPhase) -> None:
        self.phase = phase
        self.phase_history.append(phase)
        self._sync_artifacts()

    def _sync_artifacts(self) -> None:
        self.artifacts["visualization_phase"] = str(self.phase.value)
        self.artifacts["visualization_phase_history"] = [
            str(item.value) for item in self.phase_history
        ]


__all__ = [
    "ActionIntentReceipt",
    "MediaSourceRef",
    "TimelineEvent",
    "VisualScene",
    "VisualSession",
    "VisualizationPhase",
    "VisualizationSession",
]
