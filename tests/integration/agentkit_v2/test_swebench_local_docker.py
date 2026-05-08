from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from gage_eval.config.agentkit_v2 import (
    load_agentkit_v2_config_payload,
    materialize_agentkit_v2_runtime_config_payload,
    resolve_agentkit_v2_runtime_binding_specs,
)

from ._support import REPO_ROOT, build_fake_executor, read_yaml, trial_root


@pytest.mark.io
def test_swebench_local_docker_config_and_smoke_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMSTUDIO_API_KEY", "task13-secret")
    config_path = REPO_ROOT / "config/custom/swebench_pro/v2_local_docker_smoke.yaml"

    materialized = load_agentkit_v2_config_payload(config_path)
    raw_payload = read_yaml(config_path)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(
        raw_payload,
        config_path,
    )
    binding = resolve_agentkit_v2_runtime_binding_specs(
        materialized,
        runtime_config=runtime_config,
    )["swebench_dut"]

    assert binding.environment_provider == "docker"
    assert binding.scheduler_type == "framework_loop"
    assert binding.provider_config["workdir"] == "/workspace"
    assert binding.provider_config["exec_workdir"] == "/app"
    assert binding.provider_config["network_policy"] == "block"
    assert runtime_config["dut_agents"][0]["trial_policy"] == {"trials": 2}
    assert "task13-secret" not in repr(materialized)
    assert materialized["effective_config"]["backends"][0]["config"]["api_key"] == (
        "<redacted:reference:LMSTUDIO_API_KEY>"
    )

    executor, manager, scheduler, verifier = build_fake_executor(
        tmp_path,
        run_id="run-swebench-local",
        benchmark_kit_id="swebench",
        trial_policy=runtime_config["dut_agents"][0]["trial_policy"],
        environment_provider="docker",
    )
    output = asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1", "repo": "demo/repo"},
            payload={
                "execution_context": {
                    "run_id": "run-swebench-local",
                    "task_id": "swebench_pro_smoke",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    assert manager.acquired_trial_ids == ["trial_0001", "trial_0002"]
    assert scheduler.trial_ids == ["trial_0001", "trial_0002"]
    assert verifier.trial_ids == ["trial_0001", "trial_0002"]
    assert len(output["agent_eval"]["trial_results"]) == 2
    assert not (tmp_path / "run-swebench-local/samples/runtime").exists()
    for trial_id in ("trial_0001", "trial_0002"):
        root = trial_root(
            tmp_path,
            run_id="run-swebench-local",
            task_id="swebench_pro_smoke",
            sample_id="sample-1",
            trial_id=trial_id,
        )
        assert (root / "infra/trace.jsonl").is_file()
        assert (root / "infra/trial_result.json").is_file()
        assert (root / "agent/scheduler_result.json").is_file()
        assert (root / "verifier/verifier_result.json").is_file()

    aggregate = json.loads(
        (
            tmp_path
            / "run-swebench-local/artifacts/swebench_pro_smoke/sample-1/infra/trial_aggregate.json"
        ).read_text(encoding="utf-8")
    )
    assert aggregate["trial_count"] == 2
    assert aggregate["primary_trial_id"] == "trial_0001"
    sample_infra = tmp_path / "run-swebench-local/artifacts/swebench_pro_smoke/sample-1/infra"
    assert (sample_infra / "effective_config.json").is_file()
    assert (sample_infra / "sample_record.json").is_file()
    assert (sample_infra / "trial_aggregate.json").is_file()


@pytest.mark.io
def test_swebench_trials_3_fresh_verifier_acquires_environment_six_times(
    tmp_path: Path,
) -> None:
    executor, manager, scheduler, verifier = build_fake_executor(
        tmp_path,
        run_id="run-swebench-fresh-verifier",
        benchmark_kit_id="swebench",
        trial_policy={"trials": 3},
        verifier_environment_policy="fresh_from_profile",
        verifier_environment_profile_id="swebench-verifier-profile",
        environment_provider="docker",
    )

    output = asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1", "repo": "demo/repo"},
            payload={
                "execution_context": {
                    "run_id": "run-swebench-fresh-verifier",
                    "task_id": "swebench_pro_smoke",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    assert len(output["agent_eval"]["trial_results"]) == 3
    assert scheduler.trial_ids == ["trial_0001", "trial_0002", "trial_0003"]
    assert verifier.trial_ids == ["trial_0001", "trial_0002", "trial_0003"]
    assert manager.acquired_trial_ids == [
        "trial_0001",
        "trial_0001",
        "trial_0002",
        "trial_0002",
        "trial_0003",
        "trial_0003",
    ]
    scheduler_plans = manager.acquire_resource_plans[0::2]
    verifier_plans = manager.acquire_resource_plans[1::2]
    assert [plan["environment_profile"]["profile_id"] for plan in scheduler_plans] == [
        "profile",
        "profile",
        "profile",
    ]
    assert [plan["environment_profile"]["profile_id"] for plan in verifier_plans] == [
        "swebench-verifier-profile",
        "swebench-verifier-profile",
        "swebench-verifier-profile",
    ]
    assert len(manager.released_lease_ids) == 6
