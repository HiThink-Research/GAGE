from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile

import pytest
import yaml

from gage_eval.config import build_default_registry
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.registry import registry


@pytest.fixture
def fake_resolver():
    class FakeScheduler:
        def __init__(self, *, should_raise: bool) -> None:
            self.should_raise = should_raise
            self.run_calls = 0
            self.defaults = {}

        def run(self, session) -> None:
            self.run_calls += 1
            if self.should_raise:
                raise RuntimeError("scheduler failed")

    class FakeEnvironment:
        def get_active_player(self) -> str:
            return "alpha"

        def observe(self, player):
            return {"player": player}

        def apply(self, action):
            return None

        def is_terminal(self) -> bool:
            return False

    class FakeResolved:
        def __init__(self, *, should_raise: bool) -> None:
            self.resource_spec = object()
            self.scheduler = FakeScheduler(should_raise=should_raise)
            self.players = ()
            self.observation_workflow = None
            self.game_kit = type(
                "FakeGameKit",
                (),
                {"kit_id": "fake_kit", "seat_spec": {}, "defaults": {}},
            )()
            self.env_spec = type(
                "FakeEnvSpec",
                (),
                {
                    "env_id": "fake_env",
                    "defaults": {"env_factory": self._build_environment},
                },
            )()

        @staticmethod
        def _build_environment(*, sample, resolved, resources, player_specs):
            del sample, resolved, resources, player_specs
            return FakeEnvironment()

    class FakeResolver:
        def __init__(self, *, should_raise: bool) -> None:
            self.should_raise = should_raise
            self.resolve_calls = 0

        def resolve(self, sample):
            self.resolve_calls += 1
            return FakeResolved(should_raise=self.should_raise)

    def _build(*, raise_in_scheduler: bool = False):
        return FakeResolver(should_raise=raise_in_scheduler)

    return _build


@pytest.fixture
def fake_game_kit_registry():
    contracts = importlib.import_module("gage_eval.game_kits.contracts")
    registry_module = importlib.import_module("gage_eval.game_kits.registry")
    scheduler_specs = importlib.import_module("gage_eval.role.arena.schedulers.specs")
    support_workflow_module = importlib.import_module("gage_eval.role.arena.support.workflow")
    support_hooks_module = importlib.import_module("gage_eval.role.arena.support.hooks")
    support_units_module = importlib.import_module("gage_eval.role.arena.support.units")

    clone = registry.clone()
    clone.register(
        "scheduler_bindings",
        "clone_only_scheduler_v1",
        scheduler_specs.SchedulerBindingSpec(
            binding_id="clone_only_scheduler_v1",
            family="turn",
            scheduler_impl="turn",
            defaults={"max_ticks": 99},
        ),
        desc="Clone-local scheduler binding for resolver regression tests",
    )
    clone.register(
        "game_kits",
        "arc_suite_v1",
        contracts.GameKit(
            kit_id="arc_suite_v1",
            family="suite",
            scheduler_binding="turn/default",
            support_workflow="arena/default",
            observation_workflow="noop_observation_v1",
            env_catalog=(
                contracts.EnvSpec(
                    env_id="case_001",
                    kit_id="arc_suite_v1",
                    resource_spec={"env_id": "case_001", "source": "default"},
                ),
                contracts.EnvSpec(
                    env_id="case_002",
                    kit_id="arc_suite_v1",
                    resource_spec={"env_id": "case_002", "source": "sample"},
                ),
            ),
            default_env="case_001",
            seat_spec={"seats": ("alpha", "beta")},
        ),
        desc="Test suite kit with two envs",
    )
    clone.register(
        "game_kits",
        "gomoku",
        contracts.GameKit(
            kit_id="gomoku",
            family="board_game",
            scheduler_binding="turn/default",
            support_workflow="arena/default",
            observation_workflow="noop_observation_v1",
            env_catalog=(
                contracts.EnvSpec(
                    env_id="gomoku_standard",
                    kit_id="gomoku",
                    resource_spec={"env_id": "gomoku_standard"},
                ),
            ),
            default_env="gomoku_standard",
            seat_spec={"seats": ("black", "white")},
        ),
        desc="Test kit with one env",
    )
    clone.register(
        "game_kits",
        "clone_scheduler_kit_v1",
        contracts.GameKit(
            kit_id="clone_scheduler_kit_v1",
            family="suite",
            scheduler_binding="clone_only_scheduler_v1",
            support_workflow="arena/default",
            observation_workflow="noop_observation_v1",
            env_catalog=(
                contracts.EnvSpec(
                    env_id="clone_env_v1",
                    kit_id="clone_scheduler_kit_v1",
                    resource_spec={"env_id": "clone_env_v1"},
                ),
            ),
            default_env="clone_env_v1",
            seat_spec={"seats": ("alpha", "beta")},
        ),
        desc="Test kit that depends on a clone-local scheduler binding",
    )
    clone.register(
        "game_kits",
        "retro_platformer_players_v1",
        contracts.GameKit(
            kit_id="retro_platformer_players_v1",
            family="real_time_game",
            scheduler_binding="turn/default",
            support_workflow="arena/default",
            observation_workflow="noop_observation_v1",
            env_catalog=(
                contracts.EnvSpec(
                    env_id="retro_mario",
                    kit_id="retro_platformer_players_v1",
                    resource_spec={"env_id": "retro_mario"},
                ),
            ),
            default_env="retro_mario",
            seat_spec={"seats": ("player_0", "player_1", "enemy_bot")},
        ),
        desc="Realtime fixture kit for player binding tests",
    )

    def build_support_workflow():
        return support_workflow_module.GameSupportWorkflow(
            workflow_id="arena/default",
            units_by_hook={
                support_hooks_module.SupportHook.BEFORE_APPLY: [
                    support_units_module.ContinuousActionShapingUnit(low=-1.0, high=1.0)
                ]
            },
        )

    clone.register(
        "support_workflows",
        "arena/default",
        build_support_workflow,
        desc="Test support workflow with a concrete runtime unit",
    )
    yield registry_module.GameKitRegistry(registry_view=clone)


@pytest.fixture
def fake_resource_control():
    class FakeResourceControl:
        def __init__(self):
            self.release_calls = 0

        def allocate(self, resource_spec):
            return object()

        def release(self, resources):
            self.release_calls += 1

    return FakeResourceControl()


@pytest.fixture
def fake_output_writer():
    class FakeOutputWriter:
        def __init__(self) -> None:
            self.finalize_calls = 0

        def finalize(self, session):
            self.finalize_calls += 1
            return {"tick": session.tick, "step": session.step}

    return FakeOutputWriter()


@pytest.fixture
def build_arena_v2_sample():
    def _build(
        *,
        game_kit: str,
        env: str | None = None,
        players: tuple[dict[str, object], ...] = (),
        runtime_overrides: dict[str, object] | None = None,
    ) -> ArenaSample:
        return ArenaSample(
            game_kit=game_kit,
            env=env,
            players=players,
            runtime_overrides=dict(runtime_overrides or {}),
        )

    return _build


@pytest.fixture
def run_gamearena_config():
    def _run(
        config_path: str | Path,
        *,
        sample_record: dict[str, object] | None = None,
    ):
        path = Path(config_path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"Config '{path}' must be a mapping")
        record = {
            "schema_version": "gage.v1",
            "id": f"{path.stem}_sample_001",
            "messages": [{"role": "system", "content": "GameArena runtime fixture sample."}],
            "choices": [],
            **dict(sample_record or {}),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f"{path.stem}_",
            suffix=".jsonl",
            delete=False,
        ) as sample_file:
            sample_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            sample_path = Path(sample_file.name)

        try:
            payload["datasets"][0]["params"]["path"] = str(sample_path)
            config = PipelineConfig.from_dict(payload)
            registry = build_default_registry()
            runtime = build_runtime(
                config,
                registry=registry,
                resource_profile=ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]),
                trace=ObservabilityTrace(),
            )
            processed_samples: list[dict[str, object]] = []
            runtime.sample_loop.register_hook(lambda sample: processed_samples.append(sample))
            runtime.run()
            if not processed_samples:
                raise AssertionError(f"Runtime for '{path}' did not process any samples")
            processed_sample = processed_samples[0]
            predict_result = processed_sample.get("predict_result")
            if not isinstance(predict_result, list) or not predict_result:
                raise AssertionError(f"Runtime for '{path}' did not produce predict_result")
            return {
                "sample": processed_sample,
                "output": predict_result[0],
            }
        finally:
            sample_path.unlink(missing_ok=True)

    return _run
