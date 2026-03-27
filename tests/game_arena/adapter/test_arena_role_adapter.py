from __future__ import annotations

from types import MappingProxyType
from types import SimpleNamespace

from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.output.models import ArenaOutput
from gage_eval.role.adapters.arena import ArenaRoleAdapter


def test_arena_role_adapter_delegates_gamearena_payloads_to_core(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeCore:
        def run_sample(self, sample):
            captured["sample"] = sample
            return {"ok": True, "sample": sample}

    monkeypatch.setattr(
        "gage_eval.role.adapters.arena.build_gamearena_core",
        lambda **kwargs: _FakeCore(),
    )

    adapter = ArenaRoleAdapter(adapter_id="arena")
    result = adapter._invoke_sync(
        {
            "sample": {
                "game_kit": "arena/default",
                "env": None,
                "players": [{"seat": "alpha", "player_kind": "human"}],
            }
        },
        state=SimpleNamespace(),
    )

    assert result["ok"] is True
    assert captured["sample"] == ArenaSample(
        game_kit="arena/default",
        env=None,
        players=({"seat": "alpha", "player_kind": "human"},),
        runtime_overrides={},
    )


def test_arena_role_adapter_applies_configured_gamearena_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeCore:
        def run_sample(self, sample):
            captured["sample"] = sample
            return {"ok": True}

    monkeypatch.setattr(
        "gage_eval.role.adapters.arena.build_gamearena_core",
        lambda **kwargs: _FakeCore(),
    )

    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        game_kit="gomoku",
        env="gomoku_standard",
        scheduler="record_cadence/default",
        runtime_overrides={"board_size": 3, "max_steps": 9},
        players=[{"seat": "black", "player_id": "Black", "player_kind": "human"}],
    )
    adapter._invoke_sync({"sample": {"id": "sample-1", "messages": []}}, state=SimpleNamespace())

    assert captured["sample"] == ArenaSample(
        game_kit="gomoku",
        env="gomoku_standard",
        scheduler=None,
        players=({"seat": "black", "player_id": "Black", "player_kind": "human"},),
        runtime_overrides={
            "board_size": 3,
            "max_steps": 9,
            "scheduler": "record_cadence/default",
        },
    )


def test_arena_role_adapter_keeps_runtime_scheduler_override_above_adapter_default(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeCore:
        def run_sample(self, sample):
            captured["sample"] = sample
            return {"ok": True}

    monkeypatch.setattr(
        "gage_eval.role.adapters.arena.build_gamearena_core",
        lambda **kwargs: _FakeCore(),
    )

    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        game_kit="gomoku",
        env="gomoku_standard",
        scheduler="record_cadence/default",
        runtime_overrides={"board_size": 3, "max_steps": 9},
        players=[{"seat": "black", "player_id": "Black", "player_kind": "human"}],
    )
    adapter._invoke_sync(
        {
            "sample": {
                "id": "sample-1",
                "messages": [],
                "runtime_overrides": {"scheduler": "clone_only_scheduler_v1"},
            }
        },
        state=SimpleNamespace(),
    )

    assert captured["sample"] == ArenaSample(
        game_kit="gomoku",
        env="gomoku_standard",
        scheduler=None,
        players=({"seat": "black", "player_id": "Black", "player_kind": "human"},),
        runtime_overrides={
            "board_size": 3,
            "max_steps": 9,
            "scheduler": "clone_only_scheduler_v1",
        },
    )


def test_arena_role_adapter_keeps_runtime_scheduler_binding_alias_above_adapter_default(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeCore:
        def run_sample(self, sample):
            captured["sample"] = sample
            return {"ok": True}

    monkeypatch.setattr(
        "gage_eval.role.adapters.arena.build_gamearena_core",
        lambda **kwargs: _FakeCore(),
    )

    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        game_kit="gomoku",
        env="gomoku_standard",
        scheduler="record_cadence/default",
        runtime_overrides={"board_size": 3, "max_steps": 9},
        players=[{"seat": "black", "player_id": "Black", "player_kind": "human"}],
    )
    adapter._invoke_sync(
        {
            "sample": {
                "id": "sample-1",
                "messages": [],
                "runtime_overrides": {"scheduler_binding": "clone_only_scheduler_v1"},
            }
        },
        state=SimpleNamespace(),
    )

    assert captured["sample"] == ArenaSample(
        game_kit="gomoku",
        env="gomoku_standard",
        scheduler=None,
        players=({"seat": "black", "player_id": "Black", "player_kind": "human"},),
        runtime_overrides={
            "board_size": 3,
            "max_steps": 9,
            "scheduler_binding": "clone_only_scheduler_v1",
        },
    )


def test_arena_role_adapter_keeps_runtime_scheduler_binding_alias_above_adapter_runtime_default(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeCore:
        def run_sample(self, sample):
            captured["sample"] = sample
            return {"ok": True}

    monkeypatch.setattr(
        "gage_eval.role.adapters.arena.build_gamearena_core",
        lambda **kwargs: _FakeCore(),
    )

    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        game_kit="gomoku",
        env="gomoku_standard",
        runtime_overrides={
            "board_size": 3,
            "max_steps": 9,
            "scheduler": "record_cadence/default",
        },
        players=[{"seat": "black", "player_id": "Black", "player_kind": "human"}],
    )
    adapter._invoke_sync(
        {
            "sample": {
                "id": "sample-1",
                "messages": [],
                "runtime_overrides": {"scheduler_binding": "clone_only_scheduler_v1"},
            }
        },
        state=SimpleNamespace(),
    )

    assert captured["sample"] == ArenaSample(
        game_kit="gomoku",
        env="gomoku_standard",
        scheduler=None,
        players=({"seat": "black", "player_id": "Black", "player_kind": "human"},),
        runtime_overrides={
            "board_size": 3,
            "max_steps": 9,
            "scheduler_binding": "clone_only_scheduler_v1",
        },
    )


def test_arena_role_adapter_serializes_dataclass_gamearena_output(monkeypatch) -> None:
    sample = ArenaSample(
        game_kit="arena/default",
        env=None,
        players=(),
        runtime_overrides={},
    )

    class _FakeCore:
        def run_sample(self, normalized_sample):
            assert normalized_sample == sample
            return ArenaOutput(
                sample=normalized_sample,
                tick=3,
                step=7,
                arena_trace=(
                    MappingProxyType(
                        {
                            "event": "tick",
                            "payload": MappingProxyType({"value": 1}),
                        }
                    ),
                ),
            )

    monkeypatch.setattr(
        "gage_eval.role.adapters.arena.build_gamearena_core",
        lambda **kwargs: _FakeCore(),
    )

    adapter = ArenaRoleAdapter(adapter_id="arena")
    result = adapter._invoke_sync({"sample": sample}, state=SimpleNamespace())

    assert result["sample"] == {
        "game_kit": "arena/default",
        "env": None,
        "scheduler": None,
        "players": (),
        "runtime_overrides": {},
    }
    assert result["arena_trace"][0]["payload"]["value"] == 1


def test_build_gamearena_core_wires_current_registry_components(monkeypatch) -> None:
    from gage_eval.role.arena.core import bootstrap

    calls: dict[str, object] = {}

    class _FakeGameKitRegistry:
        def __init__(self, *, registry_view=None) -> None:
            calls["game_kits"] = registry_view

    class _FakeSchedulerRegistry:
        def __init__(self, *, registry_view=None) -> None:
            calls["schedulers"] = registry_view

    class _FakeObservationWorkflowRegistry:
        def __init__(self, *, registry_view=None) -> None:
            calls["observation_workflows"] = registry_view

    class _FakeSupportWorkflowRegistry:
        def __init__(self, *, registry_view=None) -> None:
            calls["support_workflows"] = registry_view

    class _FakeRuntimeBindingResolver:
        def __init__(
            self,
            *,
            game_kits,
            schedulers=None,
            observation_workflows=None,
            support_workflows=None,
        ) -> None:
            calls["resolver"] = (
                game_kits,
                schedulers,
                observation_workflows,
                support_workflows,
            )

    class _FakeResourceControl:
        def __init__(self) -> None:
            calls["resource_control"] = True

    class _FakeOutputWriter:
        def __init__(self) -> None:
            calls["output_writer"] = True

    class _FakeCore:
        def __init__(self, *, resolver, resource_control, output_writer) -> None:
            calls["core"] = (resolver, resource_control, output_writer)

    monkeypatch.setattr(bootstrap, "GameKitRegistry", _FakeGameKitRegistry)
    monkeypatch.setattr(bootstrap, "SchedulerRegistry", _FakeSchedulerRegistry)
    monkeypatch.setattr(
        bootstrap, "ObservationWorkflowRegistry", _FakeObservationWorkflowRegistry
    )
    monkeypatch.setattr(bootstrap, "SupportWorkflowRegistry", _FakeSupportWorkflowRegistry)
    monkeypatch.setattr(bootstrap, "RuntimeBindingResolver", _FakeRuntimeBindingResolver)
    monkeypatch.setattr(bootstrap, "ArenaResourceControl", _FakeResourceControl)
    monkeypatch.setattr(bootstrap, "ArenaOutputWriter", _FakeOutputWriter)
    monkeypatch.setattr(bootstrap, "GameArenaCore", _FakeCore)

    core = bootstrap.build_gamearena_core(registry_view="runtime-view")

    assert isinstance(core, _FakeCore)
    assert calls["game_kits"] == "runtime-view"
    assert calls["schedulers"] == "runtime-view"
    assert calls["observation_workflows"] == "runtime-view"
    assert calls["support_workflows"] == "runtime-view"
    assert calls["resolver"][0] is not None
    assert calls["resolver"][1] is not None
    assert calls["resolver"][2] is not None
    assert calls["resolver"][3] is not None
