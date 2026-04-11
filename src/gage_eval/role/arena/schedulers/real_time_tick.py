"""Real-time tick scheduler implementation."""

from __future__ import annotations

import time
from typing import Mapping

from gage_eval.role.arena.schedulers.base import Scheduler


class RealTimeTickScheduler(Scheduler):
    family = "real_time_tick"

    def __init__(
        self,
        *,
        binding_id: str,
        family: str | None = None,
        defaults: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(binding_id=binding_id, family=family, defaults=defaults)

    def run(self, session) -> None:
        tick_ms = _resolve_scheduler_owned_tick_interval_ms(session)
        if tick_ms is not None:
            _run_scheduler_owned_realtime_loop(session, tick_ms=tick_ms)
            return
        while not session.should_stop():
            observation = session.observe()
            decision = session.decide_current_player(observation)
            if decision is not None:
                session.apply(decision)
                session.advance()
            else:
                session.advance(decision_taken=False)


def _run_scheduler_owned_realtime_loop(session, *, tick_ms: int) -> None:
    tick_interval_s = float(tick_ms) / 1000.0
    while not session.should_stop():
        tick_started = _monotonic_seconds()
        observation = session.observe()
        decisions = _drain_scheduler_owned_decisions(session, observation)
        if decisions:
            for decision in decisions:
                session.apply(decision)
                session.advance(decision_taken=True, tick_delta=0)
                if getattr(session, "final_result", None) is not None:
                    break
            session.advance(decision_taken=False, tick_delta=1)
        else:
            session.advance(decision_taken=False)
        capture_output_tick = getattr(session, "capture_output_tick", None)
        if callable(capture_output_tick):
            capture_output_tick()
        elapsed_ms = (_monotonic_seconds() - tick_started) * 1000.0
        record_metrics = getattr(session, "record_realtime_tick_metrics", None)
        if callable(record_metrics):
            record_metrics(
                tick_interval_ms=tick_ms,
                tick_elapsed_ms=elapsed_ms,
            )
        remaining_s = tick_interval_s - ((_monotonic_seconds() - tick_started))
        if remaining_s > 0.0:
            time.sleep(remaining_s)


def _drain_scheduler_owned_decisions(session, observation) -> list[object]:
    drain_actions = getattr(session, "drain_scheduler_owned_current_player_actions", None)
    if not callable(drain_actions):
        decision = session.decide_current_player(observation)
        return [] if decision is None else [decision]

    max_actions_per_tick = 1
    max_actions_resolver = getattr(session, "max_scheduler_owned_actions_per_tick", None)
    if callable(max_actions_resolver):
        max_actions_per_tick = max(1, int(max_actions_resolver()))
    return list(drain_actions(observation, max_items=max_actions_per_tick))[:max_actions_per_tick]


def _resolve_scheduler_owned_tick_interval_ms(session) -> int | None:
    profile = getattr(session, "runtime_profile", None)
    if profile is None:
        return None
    uses_scheduler_owned = getattr(profile, "uses_scheduler_owned_human_realtime", None)
    if callable(uses_scheduler_owned):
        if not bool(uses_scheduler_owned()):
            return None
    elif not bool(getattr(profile, "scheduler_owns_realtime_clock", False)):
        return None
    value = getattr(profile, "tick_interval_ms", None)
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    if normalized <= 0:
        return None
    return normalized


def _monotonic_seconds() -> float:
    monotonic = getattr(time, "monotonic", None)
    if callable(monotonic):
        return float(monotonic())
    return float(__import__("time").monotonic())
