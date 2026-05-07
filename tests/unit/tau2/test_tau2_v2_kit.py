from __future__ import annotations

from types import SimpleNamespace

from gage_eval.agent_eval_kits.common import validate_benchmark_kit_entry
from gage_eval.agent_eval_kits.tau2.kit import load_kit
from gage_eval.agent_eval_kits.tau2.sub_workflows.framework_loop import build_workflow_bundle


def test_tau2_kit_declares_v2_environment_and_verifier_contracts() -> None:
    kit = load_kit()

    assert kit.default_environment_provider == "local_process"
    assert kit.default_environment_profile_by_provider["local_process"] == "tau2-local-process"
    assert kit.default_environment_profile_by_provider["e2b"] == "tau2-e2b-wrapper"
    assert kit.environment_profiles["tau2-local-process"]["asset_dir"].endswith(
        "agent_eval_kits/tau2/environment/local_process"
    )
    assert kit.environment_profiles["tau2-e2b-wrapper"]["asset_dir"].endswith(
        "agent_eval_kits/tau2/environment/e2b"
    )
    assert kit.environment_profiles["tau2-e2b-wrapper"]["config"]["template_id"] == "gage-tau2-wrapper"
    assert kit.verifier_environment_policy == "reuse"
    assert kit.verifier_environment_profile_id is None
    assert kit.verifier_adapter_factory is not None
    assert kit.artifact_manifest_factory is not None
    assert kit.build_verifier_adapter().__class__.__module__ == (
        "gage_eval.agent_eval_kits.tau2.judge.adapters"
    )
    assert validate_benchmark_kit_entry(kit) is kit


def test_tau2_framework_loop_declares_respond_as_required_tool() -> None:
    workflow = build_workflow_bundle()

    loop_inputs = workflow.build_loop_inputs(
        session=SimpleNamespace(prompt_context={}, benchmark_state={}),
        sample={"messages": [{"role": "user", "content": "hello"}]},
        payload={},
    )

    assert loop_inputs["required_tool"] == "respond"
