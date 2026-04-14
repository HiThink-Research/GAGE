from __future__ import annotations

from types import SimpleNamespace

import gage_eval.role.arena.schedulers.real_time_tick as real_time_tick_module
import pytest

from gage_eval.role.arena.core.game_session import (
    GameSession,
    _resolve_max_steps,
    _resolve_max_tick_budget,
)
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.schedulers.real_time_tick import RealTimeTickScheduler
from gage_eval.role.arena.types import ArenaAction, ArenaObservation
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
                legal_actions_items=("noop", "issue_command"),
                view_text="realtime frame",
                board_text="realtime frame",
            )

        def consume_session_progress_delta(self) -> int:
            return 1

        def is_terminal(self) -> bool:
            return False

    return GameSession(
        sample=ArenaSample(game_kit="sample_realtime", env="queued_command"),
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
    sample = ArenaSample(game_kit="sample_realtime", env="queued_command")
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
        game_kit="sample_realtime",
        env="queued_command",
        runtime_overrides={"max_decisions": 7200},
    )
    resolved = SimpleNamespace(scheduler=SimpleNamespace(defaults={"max_ticks": 256}))

    assert _resolve_max_steps(sample=sample, resolved=resolved) == 7200
    assert _resolve_max_tick_budget(sample=sample, resolved=resolved) is None


def test_resolve_max_tick_budget_keeps_explicit_max_ticks_even_with_max_decisions() -> None:
    sample = ArenaSample(
        game_kit="sample_realtime",
        env="queued_command",
        runtime_overrides={"max_decisions": 7200, "max_ticks": 4096},
    )
    resolved = SimpleNamespace(scheduler=SimpleNamespace(defaults={"max_ticks": 256}))

    assert _resolve_max_steps(sample=sample, resolved=resolved) == 7200
    assert _resolve_max_tick_budget(sample=sample, resolved=resolved) == 4096


def test_scheduler_owned_realtime_max_ticks_budget_counts_loops_not_drained_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RealtimeQueuedCommandPlayer:
        player_id = "player_0"
        display_name = "player_0"
        seat = "player_0"
        player_kind = "human"
        metadata: dict[str, object] = {}

        def __init__(self) -> None:
            self.drain_calls = 0
            self._queued_batches = [
                [
                    ArenaAction(player="player_0", move="issue_command", raw="issue_command"),
                    ArenaAction(player="player_0", move="followup_command", raw="followup_command"),
                ],
                [],
            ]

        def drain_scheduler_owned_actions(
            self,
            observation: ArenaObservation,
            *,
            max_items: int,
        ) -> list[ArenaAction]:
            del observation
            self.drain_calls += 1
            batch = self._queued_batches[min(self.drain_calls - 1, len(self._queued_batches) - 1)]
            return list(batch[:max_items])

    class FakeEnvironment:
        def __init__(self) -> None:
            self.observe_calls = 0
            self.applied_moves: list[str] = []

        def get_active_player(self) -> str:
            return "player_0"

        def observe(self, player_id: str) -> ArenaObservation:
            self.observe_calls += 1
            return ArenaObservation(
                board_text="realtime frame",
                legal_moves=("issue_command", "followup_command"),
                active_player=player_id,
            )

        def apply(self, action: ArenaAction) -> None:
            self.applied_moves.append(action.move)
            return None

        def is_terminal(self) -> bool:
            return False

    current_time = {"value": 100.0}

    monkeypatch.setattr(
        real_time_tick_module,
        "time",
        SimpleNamespace(
            sleep=lambda delay: current_time.__setitem__("value", current_time["value"] + delay),
            monotonic=lambda: current_time["value"],
        ),
        raising=False,
    )

    player = RealtimeQueuedCommandPlayer()
    environment = FakeEnvironment()
    session = GameSession(
        sample=ArenaSample(game_kit="sample_realtime", env="queued_command"),
        environment=environment,
        player_specs=(player,),
        max_steps=4,
        max_ticks=2,
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
                max_commands_per_tick=2,
            ),
        ),
    )

    RealTimeTickScheduler(binding_id="real_time_tick/default").run(session)

    assert player.drain_calls == 2
    assert environment.observe_calls == 2
    assert environment.applied_moves == ["issue_command", "followup_command"]
    assert session.tick == 2
    assert session.step == 2
    assert session.final_result is not None
    assert session.final_result["result"] == "max_ticks"


def test_real_time_tick_scheduler_drives_env_forward_when_no_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoInputPlayer:
        player_id = "player_0"
        display_name = "player_0"
        seat = "player_0"
        player_kind = "human"
        metadata: dict[str, object] = {}

        def drain_scheduler_owned_actions(
            self,
            observation: ArenaObservation,
            *,
            max_items: int,
        ) -> list[ArenaAction]:
            del observation, max_items
            return []

    class FakeEnvironment:
        def __init__(self) -> None:
            self.observe_calls = 0
            self.tick_idle_calls: list[int] = []

        def get_active_player(self) -> str:
            return "player_0"

        def observe(self, player_id: str) -> ArenaObservation:
            self.observe_calls += 1
            return ArenaObservation(
                board_text="realtime frame",
                legal_moves=("noop",),
                active_player=player_id,
            )

        def tick_idle(self, *, frames: int, move: str) -> None:
            assert move == "noop"
            self.tick_idle_calls.append(frames)
            return None

        def is_terminal(self) -> bool:
            return False

    current_time = {"value": 100.0}
    monkeypatch.setattr(
        real_time_tick_module,
        "time",
        SimpleNamespace(
            sleep=lambda delay: current_time.__setitem__("value", current_time["value"] + delay),
            monotonic=lambda: current_time["value"],
        ),
        raising=False,
    )

    environment = FakeEnvironment()
    session = GameSession(
        sample=ArenaSample(game_kit="retro_platformer", env="retro_mario"),
        environment=environment,
        player_specs=(NoInputPlayer(),),
        max_steps=4,
        max_ticks=4,
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=16,
            pure_human_realtime=True,
            scheduler_owns_realtime_clock=True,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=True,
            realtime_human_control=RealtimeHumanControlProfile(
                mode="scheduler_owned_human_realtime",
                activation_scope="pure_human_only",
                input_model="continuous_state",
                tick_interval_ms=16,
                input_transport="realtime_ws",
                fallback_move="noop",
            ),
        ),
    )

    RealTimeTickScheduler(binding_id="real_time_tick/default").run(session)

    assert environment.tick_idle_calls == [1, 1, 1, 1]
    assert environment.observe_calls == 1
    assert session.tick == 4
    assert session.step == 0
    assert session.final_result is not None
    assert session.final_result["result"] == "max_ticks"


def test_real_time_tick_scheduler_idles_until_first_human_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DelayedInputPlayer:
        player_id = "player_0"
        display_name = "player_0"
        seat = "player_0"
        player_kind = "human"
        metadata: dict[str, object] = {}

        def __init__(self) -> None:
            self.drain_calls = 0

        def drain_scheduler_owned_actions(
            self,
            observation: ArenaObservation,
            *,
            max_items: int,
        ) -> list[ArenaAction]:
            del observation, max_items
            self.drain_calls += 1
            if self.drain_calls == 6:
                return [ArenaAction(player="player_0", move="right", raw="right")]
            return []

    class FakeEnvironment:
        def __init__(self) -> None:
            self.events: list[str] = []

        def get_active_player(self) -> str:
            return "player_0"

        def observe(self, player_id: str) -> ArenaObservation:
            return ArenaObservation(
                board_text="realtime frame",
                legal_moves=("noop", "right"),
                active_player=player_id,
            )

        def apply(self, action: ArenaAction) -> None:
            self.events.append(f"apply:{action.move}")
            return None

        def tick_idle(self, *, frames: int, move: str) -> None:
            assert frames == 1
            assert move == "noop"
            self.events.append("idle")
            return None

        def is_terminal(self) -> bool:
            return False

    current_time = {"value": 200.0}
    monkeypatch.setattr(
        real_time_tick_module,
        "time",
        SimpleNamespace(
            sleep=lambda delay: current_time.__setitem__("value", current_time["value"] + delay),
            monotonic=lambda: current_time["value"],
        ),
        raising=False,
    )

    player = DelayedInputPlayer()
    environment = FakeEnvironment()
    session = GameSession(
        sample=ArenaSample(game_kit="retro_platformer", env="retro_mario"),
        environment=environment,
        player_specs=(player,),
        max_steps=4,
        max_ticks=7,
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=16,
            pure_human_realtime=True,
            scheduler_owns_realtime_clock=True,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=True,
            realtime_human_control=RealtimeHumanControlProfile(
                mode="scheduler_owned_human_realtime",
                activation_scope="pure_human_only",
                input_model="continuous_state",
                tick_interval_ms=16,
                input_transport="realtime_ws",
                fallback_move="noop",
            ),
        ),
    )

    RealTimeTickScheduler(binding_id="real_time_tick/default").run(session)

    assert player.drain_calls == 7
    assert environment.events == ["idle", "idle", "idle", "idle", "idle", "apply:right", "idle"]
    assert session.tick == 7
    assert session.step == 1
    assert session.final_result is not None
    assert session.final_result["result"] == "max_ticks"
