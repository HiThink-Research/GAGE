"""Record-cadence scheduler implementation."""

from __future__ import annotations

from typing import Mapping

from gage_eval.role.arena.schedulers.base import Scheduler


class RecordCadenceScheduler(Scheduler):
    family = "record_cadence"

    def __init__(
        self,
        *,
        binding_id: str,
        family: str | None = None,
        defaults: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(binding_id=binding_id, family=family, defaults=defaults)

    def run(self, session) -> None:
        while not session.should_stop():
            observation = session.observe()
            decision = session.decide_current_player(observation)
            if decision is None:
                if session.should_stop():
                    break
                raise RuntimeError("RecordCadenceScheduler received no decision for a live turn")
            session.apply(decision)
            session.capture_output_tick()
            session.advance()
