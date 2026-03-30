from __future__ import annotations

import pytest

from gage_eval.role.arena.resources.control import ArenaResourceControl
from gage_eval.role.arena.resources.specs import ArenaResources


@pytest.fixture
def fake_process_handle():
    class FakeProcessHandle:
        def __init__(self) -> None:
            self.closed = False
            self.terminated = False
            self.reap_calls = 0

        def close(self) -> None:
            self.closed = True

        def terminate(self) -> None:
            self.terminated = True

        def reap(self) -> None:
            self.reap_calls += 1

    return FakeProcessHandle()


def test_arena_resource_control_release_closes_terminates_and_reaps_runtime(
    fake_process_handle,
) -> None:
    resources = ArenaResources(game_runtime=fake_process_handle)

    ArenaResourceControl().release(resources)

    assert fake_process_handle.closed is True
    assert fake_process_handle.terminated is True
    assert fake_process_handle.reap_calls == 1


def test_arena_resource_control_allocate_tracks_resource_categories_and_phase() -> None:
    resources = ArenaResourceControl().allocate(
        {"env_id": "gomoku_standard", "family": "gomoku"}
    )

    assert resources.resource_categories == (
        "game_runtime_resource",
        "game_bridge_resource",
    )
    assert resources.lifecycle_phase == "allocated"


def test_arena_resource_control_release_still_terminates_and_reaps_if_close_fails() -> None:
    class CloseFailingHandle:
        def __init__(self) -> None:
            self.closed = False
            self.terminated = False
            self.reap_calls = 0

        def close(self) -> None:
            self.closed = True
            raise RuntimeError("close failed")

        def terminate(self) -> None:
            self.terminated = True

        def reap(self) -> None:
            self.reap_calls += 1

    handle = CloseFailingHandle()
    resources = ArenaResources(game_runtime=handle)

    with pytest.raises(Exception, match="close failed") as exc_info:
        ArenaResourceControl().release(resources)

    error = exc_info.value
    assert getattr(error, "error_code", None) == "resource_lifecycle_error"
    assert error.errors[0]["operation"] == "close"
    assert error.errors[0]["resource_category"] == "game_runtime_resource"
    assert handle.terminated is True
    assert handle.reap_calls == 1
