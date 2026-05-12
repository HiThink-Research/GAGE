from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from gage_eval.agent_eval_kits.swebench.judge.adapters import SwebenchVerifierAdapter
from gage_eval.agent_runtime.resources.manager import RuntimeResourceManager
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.verifier.contracts import VerifierInput
from gage_eval.environment.contracts import ExecResult

from ._support import REPO_ROOT, load_lowered_pipeline_config, role_adapter_by_id


DIFF = "diff --git a/foo b/foo\n--- a/foo\n+++ b/foo\n@@ -1 +1 @@\n-old\n+new\n"
TARGET_TEST = "tests/test_fix.py::test_fix"


class _FakeVerifierEnvironment:
    env_id = "env-1"
    name = "fake"
    metadata = {}
    capabilities = {}

    def __init__(self, *, provider: str) -> None:
        self.provider = provider
        self.writes: dict[str, bytes] = {}

    async def write_file(self, path: str, content: bytes | str) -> None:
        self.writes[path] = content.encode("utf-8") if isinstance(content, str) else content

    async def read_file(self, path: str, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
        del max_bytes
        if path == "/workspace/output.json":
            return json.dumps({"tests": [{"name": TARGET_TEST, "status": "PASSED"}]}).encode()
        try:
            return self.writes[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc

    async def exec(self, command: str, **kwargs: Any) -> ExecResult:
        del kwargs
        return ExecResult(command=command, exit_code=0)


@pytest.mark.io
def test_swebench_e2b_wrapper_config_materializes_to_e2b_profile() -> None:
    config_path = REPO_ROOT / "config/custom/swebench_pro/v2_e2b_wrapper_smoke.yaml"

    materialized = load_lowered_pipeline_config(config_path)
    role_adapter = role_adapter_by_id(materialized, "swebench_dut")
    params = role_adapter["params"]

    assert role_adapter["agent_runtime_id"] == "swebench_framework_loop"
    assert params["environment_profile"]["provider"] == "e2b"
    assert params["environment_profile"]["profile_id"] == "swebench-e2b-wrapper"
    assert params["provider_config"]["template_id"] == "gage-swebench-pro-wrapper"
    assert params["provider_config"]["network_policy"] == "block"


@pytest.mark.io
def test_docker_and_e2b_verifier_result_scoring_field_sets_match(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "run_scripts"
    _write_scripts(scripts_dir)

    docker_payload = _run_adapter_payload(provider="docker", scripts_dir=scripts_dir)
    e2b_payload = _run_adapter_payload(provider="e2b", scripts_dir=scripts_dir)

    assert set(_normalize_scoring_payload(docker_payload)) == set(_normalize_scoring_payload(e2b_payload))
    assert set(docker_payload["metric"]) == set(e2b_payload["metric"])
    assert docker_payload["metric"]["score"] == e2b_payload["metric"]["score"] == 1.0


@pytest.mark.live
def test_live_e2b_wrapper_starts_and_transfers_files(tmp_path: Path) -> None:
    if not os.getenv("E2B_API_KEY"):
        pytest.skip("E2B_API_KEY is required for live E2B wrapper smoke")
    config_path = REPO_ROOT / "config/custom/swebench_pro/v2_e2b_wrapper_smoke.yaml"
    materialized = load_lowered_pipeline_config(config_path)
    params = role_adapter_by_id(materialized, "swebench_dut")["params"]
    session = AgentRuntimeSession(
        session_id="live-e2b-wrapper-session",
        run_id="live-e2b-wrapper-run",
        task_id="live-e2b-wrapper-task",
        sample_id="live-e2b-wrapper-smoke",
        benchmark_kit_id="swebench",
        scheduler_type="framework_loop",
    )
    manager = RuntimeResourceManager()
    lease_binding = None
    try:
        lease_binding = manager.acquire(
            session,
            resource_plan={
                "resource_kind": params["environment_profile"]["provider"],
                "environment_profile": params["environment_profile"],
                "provider_config": params["provider_config"],
            },
        )
        assert lease_binding.sandbox_provider is None
        environment_lease = lease_binding.environment_lease
        assert environment_lease is not None
        result = asyncio.run(
            environment_lease.exec(
                "test -d /workspace && test -d /logs/verifier && echo wrapper-ready",
                timeout_s=60,
            )
        )
        assert result.exit_code == 0, result.stderr
        assert "wrapper-ready" in result.stdout

        source = tmp_path / "live-payload.txt"
        target = tmp_path / "live-downloaded.txt"
        source.write_text("live wrapper transfer\n", encoding="utf-8")
        asyncio.run(environment_lease.upload_file(source, "/workspace/live-payload.txt"))
        asyncio.run(environment_lease.download_file("/workspace/live-payload.txt", target))
        assert target.read_text(encoding="utf-8") == "live wrapper transfer\n"
    finally:
        if lease_binding is not None:
            manager.release(lease_binding)


def _write_scripts(root: Path) -> None:
    run_dir = root / "instance_1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (run_dir / "parser.py").write_text(
        "import json, sys\n"
        f"json.dump({{'tests': [{{'name': {TARGET_TEST!r}, 'status': 'PASSED'}}]}}, open(sys.argv[3], 'w'))\n",
        encoding="utf-8",
    )


def _run_adapter_payload(*, provider: str, scripts_dir: Path) -> dict[str, Any]:
    result = SwebenchVerifierAdapter(scripts_dir=str(scripts_dir)).run(
        VerifierInput(
            benchmark_kit_id="swebench",
            scheduler_type="framework_loop",
            sample_id="instance_1",
            sample={
                "id": "instance_1",
                "metadata": {
                    "instance_id": "instance_1",
                    "repo": "repo/name",
                    "base_commit": "abc123",
                    "fail_to_pass": [TARGET_TEST],
                    "pass_to_pass": [],
                },
            },
            scheduler_result={"agent_output": {"answer": DIFF}},
            runtime_context={"environment": _FakeVerifierEnvironment(provider=provider)},
            verifier_resources={},
        )
    )
    return dict(result.payload)


def _normalize_scoring_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key
        in {
            "resolved",
            "score",
            "failure_reason",
            "tests",
            "fail_tests",
            "pass_tests",
            "status",
            "failure_category",
            "metric",
            "artifact_refs",
        }
    }
