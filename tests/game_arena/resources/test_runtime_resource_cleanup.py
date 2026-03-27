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

    with pytest.raises(RuntimeError, match="close failed"):
        ArenaResourceControl().release(resources)

    assert handle.terminated is True
    assert handle.reap_calls == 1
