from __future__ import annotations

import pytest

from gage_eval.role.arena.resources.visualization import (
    VisualizationPhase,
    VisualizationSession,
)


@pytest.fixture
def fake_display():
    class FakeDisplay:
        def __init__(self) -> None:
            self.closed_inputs = 0

        def close_inputs(self) -> None:
            self.closed_inputs += 1

    return FakeDisplay()


@pytest.fixture
def fake_replay_viewer():
    class FakeReplayViewer:
        def __init__(self) -> None:
            self.loaded = []

        def load(self, replay_uri: str) -> None:
            self.loaded.append(replay_uri)

    return FakeReplayViewer()


def test_visualization_session_switches_live_to_replay(
    fake_display,
    fake_replay_viewer,
) -> None:
    session = VisualizationSession(
        phase=VisualizationPhase.LIVE,
        display=fake_display,
        replay_viewer=fake_replay_viewer,
    )
    session.switch_to_replay("replay://mario/sample-1")

    assert session.phase is VisualizationPhase.REPLAY_READY
    assert fake_display.closed_inputs == 1
    assert fake_replay_viewer.loaded == ["replay://mario/sample-1"]
