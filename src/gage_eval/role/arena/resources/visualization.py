"""Compatibility wrapper for the legacy visualization session API.

New JSON-safe contracts live in :mod:`gage_eval.role.arena.visualization.contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    REPLAY_READY = "replay_ready"


@dataclass
class VisualizationSession:
    phase: VisualizationPhase
    display: object
    replay_viewer: object

    def switch_to_replay(self, replay_uri: str) -> None:
        self.display.close_inputs()
        self.replay_viewer.load(replay_uri)
        self.phase = VisualizationPhase.REPLAY_READY


__all__ = [
    "ActionIntentReceipt",
    "MediaSourceRef",
    "TimelineEvent",
    "VisualScene",
    "VisualSession",
    "VisualizationPhase",
    "VisualizationSession",
]
