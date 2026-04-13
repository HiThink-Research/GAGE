from __future__ import annotations

from gage_eval.role.arena.schedulers.record_cadence import RecordCadenceScheduler


def test_record_cadence_scheduler_captures_each_tick() -> None:
    class FakeSession:
        def __init__(self, *, stop_after: int) -> None:
            self.stop_after = stop_after
            self.tick = 0
            self.capture_calls = 0
            self.applied_actions: list[dict[str, int]] = []

        def should_stop(self) -> bool:
            return self.tick >= self.stop_after

        def observe(self) -> dict[str, int]:
            return {"tick": self.tick}

        def decide_current_player(self, observation: dict[str, int]) -> dict[str, int]:
            return {"action": observation["tick"]}

        def apply(self, decision: dict[str, int]) -> None:
            self.applied_actions.append(decision)

        def capture_output_tick(self) -> None:
            self.capture_calls += 1

        def advance(self) -> None:
            self.tick += 1

    scheduler = RecordCadenceScheduler(binding_id="record_cadence/default")
    session = FakeSession(stop_after=3)

    scheduler.run(session)

    assert session.capture_calls == 3
    assert session.applied_actions == [{"action": 0}, {"action": 1}, {"action": 2}]
