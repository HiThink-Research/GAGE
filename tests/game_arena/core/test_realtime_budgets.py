from __future__ import annotations

from types import SimpleNamespace

from gage_eval.role.arena.core.game_session import (
    GameSession,
    _resolve_max_steps,
    _resolve_max_tick_budget,
)
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.game_kits.contracts import (
    HumanRealtimeInputProfile,
    RealtimeHumanControlProfile,
    ResolvedRuntimeProfile,
)


def _build_idle_scheduler_owned_session(*, max_steps: int, max_ticks: int | None) -> GameSession:
    class FakeEnvironment:
        def get_active_player(self) -> str:
            return "player_0"

        def observe(self, player_id: str) -> object:
            return SimpleNamespace(
                active_player=player_id,
                legal_actions_items=("noop", "bridge_input"),
                view_text="openra frame",
                board_text="openra frame",
            )

        def consume_session_progress_delta(self) -> int:
            return 1

        def is_terminal(self) -> bool:
            return False

    return GameSession(
        sample=ArenaSample(game_kit="openra", env="ra_skirmish_1v1"),
        environment=FakeEnvironment(),
        player_specs=(SimpleNamespace(player_id="player_0", player_kind="human"),),
        max_steps=max_steps,
        max_ticks=max_ticks,
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=50,
            pure_human_realtime=True,
            scheduler_owns_realtime_clock=True,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=True,
            human_realtime_inputs=(
                HumanRealtimeInputProfile(
                    player_id="player_0",
                    semantics="queued_command",
                    tick_interval_ms=50,
                ),
            ),
            realtime_human_control=RealtimeHumanControlProfile(
                mode="scheduler_owned_human_realtime",
                activation_scope="pure_human_only",
                input_model="queued_command",
                tick_interval_ms=50,
                input_transport="realtime_ws",
                frame_output_hz=20,
                artifact_sampling_mode="async_decimated_live",
                fallback_move="noop",
            ),
        ),
    )


def test_game_session_inherits_scheduler_default_max_ticks_as_idle_tick_cap_when_no_explicit_decision_budget() -> None:
    sample = ArenaSample(game_kit="openra", env="ra_skirmish_1v1")
    resolved = SimpleNamespace(scheduler=SimpleNamespace(defaults={"max_ticks": 2}))

    session = _build_idle_scheduler_owned_session(
        max_steps=_resolve_max_steps(sample=sample, resolved=resolved),
        max_ticks=_resolve_max_tick_budget(sample=sample, resolved=resolved),
    )

    session.observe()
    session.advance()

    assert session.should_stop() is False

    session.observe()
    session.advance()

    assert session.should_stop() is True
    assert session.final_result is not None
    assert session.final_result["result"] == "max_ticks"


def test_resolve_max_tick_budget_does_not_inherit_scheduler_default_when_max_decisions_explicit() -> None:
    sample = ArenaSample(
        game_kit="openra",
        env="ra_skirmish_1v1",
        runtime_overrides={"max_decisions": 7200},
    )
    resolved = SimpleNamespace(scheduler=SimpleNamespace(defaults={"max_ticks": 256}))

    assert _resolve_max_steps(sample=sample, resolved=resolved) == 7200
    assert _resolve_max_tick_budget(sample=sample, resolved=resolved) is None


def test_resolve_max_tick_budget_keeps_explicit_max_ticks_even_with_max_decisions() -> None:
    sample = ArenaSample(
        game_kit="openra",
        env="ra_skirmish_1v1",
        runtime_overrides={"max_decisions": 7200, "max_ticks": 4096},
    )
    resolved = SimpleNamespace(scheduler=SimpleNamespace(defaults={"max_ticks": 256}))

    assert _resolve_max_steps(sample=sample, resolved=resolved) == 7200
    assert _resolve_max_tick_budget(sample=sample, resolved=resolved) == 4096
