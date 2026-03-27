from __future__ import annotations

import importlib

from gage_eval.registry import registry


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
