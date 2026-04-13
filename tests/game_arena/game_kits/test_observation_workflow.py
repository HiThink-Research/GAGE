from __future__ import annotations

import importlib

from gage_eval.registry import registry


def test_game_kit_contract_preserves_explicit_optional_runtime_refs() -> None:
    contracts = importlib.import_module("gage_eval.game_kits.contracts")

    env = contracts.EnvSpec(
        env_id="env_v1",
        kit_id="kit_v1",
        observation_workflow="obs/env",
        game_content_refs={"env_ref": "content/env"},
        runtime_binding_policy="policy/env",
        game_display="display/env",
        replay_viewer="viewer/env",
        parser="parser/env",
        renderer="renderer/env",
        replay_policy="replay/env",
        input_mapper="input/env",
    )
    kit = contracts.GameKit(
        kit_id="kit_v1",
        family="board_game",
        scheduler_binding="turn/default",
        observation_workflow="obs/kit",
        env_catalog=(env,),
        default_env="env_v1",
        support_workflow="arena/default",
        visualization_spec="display/kit",
        content_asset="content/kit",
        game_content_refs={"kit_ref": "content/kit"},
        runtime_binding_policy="policy/kit",
        game_display="display/kit",
        replay_viewer="viewer/kit",
        parser="parser/kit",
        renderer="renderer/kit",
        replay_policy="replay/kit",
        input_mapper="input/kit",
    )

    assert kit.runtime_binding_policy == "policy/kit"
    assert kit.game_display == "display/kit"
    assert kit.replay_viewer == "viewer/kit"
    assert kit.parser == "parser/kit"
    assert kit.renderer == "renderer/kit"
    assert kit.replay_policy == "replay/kit"
    assert kit.input_mapper == "input/kit"
    assert kit.game_content_refs == {"kit_ref": "content/kit"}
    assert env.runtime_binding_policy == "policy/env"
    assert env.game_display == "display/env"
    assert env.replay_viewer == "viewer/env"
    assert env.parser == "parser/env"
    assert env.renderer == "renderer/env"
    assert env.replay_policy == "replay/env"
    assert env.input_mapper == "input/env"
    assert env.game_content_refs == {"env_ref": "content/env"}


def test_observation_workflow_registry_roundtrips_custom_builder() -> None:
    observation_module = importlib.import_module("gage_eval.game_kits.observation")
    registry_module = importlib.import_module("gage_eval.game_kits.registry")

    clone = registry.clone()

    def build_custom_workflow():
        return observation_module.ObservationWorkflow(
            workflow_id="custom_obs_v1",
            defaults={"source": "roundtrip"},
        )

    with registry.route_to(clone):
        registry.register(
            "observation_workflows",
            "custom_obs_v1",
            build_custom_workflow,
            desc="Custom observation workflow for tests",
        )
        workflow = registry_module.ObservationWorkflowRegistry(registry_view=clone).build("custom_obs_v1")

    assert workflow.workflow_id == "custom_obs_v1"
    assert workflow.defaults == {"source": "roundtrip"}
    assert workflow.build({"tick": 1}, object()) == {"tick": 1}


def test_observation_workflow_registry_builds_noop_workflow() -> None:
    registry_module = importlib.import_module("gage_eval.game_kits.registry")

    workflow = registry_module.ObservationWorkflowRegistry().build("noop_observation_v1")

    assert workflow.workflow_id == "noop_observation_v1"
    assert workflow.build({"message": "ok"}, object()) == {"message": "ok"}
