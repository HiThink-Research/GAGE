from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from gage_eval.agent_eval_kits.swebench.judge.adapters import SwebenchVerifierAdapter
from gage_eval.agent_eval_kits.swebench.kit import load_kit
from gage_eval.agent_runtime.resources.manager import RuntimeResourceManager
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.verifier.contracts import VerifierInput
from gage_eval.config.agentkit_v2 import (
    build_agentkit_v2_runtime_bindings,
    materialize_agentkit_v2_config_payload,
    materialize_agentkit_v2_runtime_config_payload,
    resolve_agentkit_v2_runtime_binding_specs,
)
from gage_eval.environment import EnvironmentResources
from gage_eval.environment.contracts import ExecResult
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.providers.e2b import E2BEnvironmentProvider


DIFF = "diff --git a/foo b/foo\n--- a/foo\n+++ b/foo\n@@ -1 +1 @@\n-old\n+new\n"
TARGET_TEST = "tests/test_fix.py::test_fix"


class _FakeFiles:
    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}

    def write(self, path: str, payload: bytes, **kwargs: Any) -> None:
        del kwargs
        self._files[path] = bytes(payload)

    def read(self, path: str, **kwargs: Any) -> bytes:
        del kwargs
        try:
            return self._files[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc

    def make_dir(self, path: str, **kwargs: Any) -> None:
        del path, kwargs


class _FakeSandbox:
    def __init__(self) -> None:
        self.files = _FakeFiles()

    def kill(self) -> None:
        return None


class _FakeE2BClient:
    def __init__(self) -> None:
        self.create_kwargs: dict[str, Any] | None = None
        self.sandbox = _FakeSandbox()

    def create_sandbox(self, **kwargs: Any) -> _FakeSandbox:
        self.create_kwargs = dict(kwargs)
        return self.sandbox


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


def test_swebench_kit_declares_e2b_wrapper_profile_and_reuses_verifier_adapter(tmp_path: Path) -> None:
    kit = load_kit()

    assert kit.default_environment_provider == "docker"
    assert kit.default_environment_profile_by_provider["docker"] == "swebench_runtime"
    assert kit.default_environment_profile_by_provider["e2b"] == "swebench-e2b-wrapper"

    profile = kit.environment_profiles["swebench-e2b-wrapper"]
    assert profile["asset_dir"].endswith("environment/e2b")
    assert Path(profile["asset_dir"]).name == "e2b"
    assert profile["config"]["template_id"] == "gage-swebench-pro-wrapper"
    assert profile["capabilities"]["supports_upload_download"] is True
    assert profile["capabilities"]["supports_privileged_dind"] is False
    assert kit.build_verifier_adapter().__class__.__module__ == "gage_eval.agent_eval_kits.swebench.judge.adapters"

    client = _FakeE2BClient()
    provider = E2BEnvironmentProvider(client=client)
    environment = asyncio.run(
        provider.create(
            kit_id="swebench",
            provider="e2b",
            profile_id="swebench-e2b-wrapper",
            profile=EnvironmentProfile(
                profile_id="swebench-e2b-wrapper",
                provider="e2b",
                config=profile["config"],
            ),
            provider_config={},
            resources=EnvironmentResources(network_policy="block"),
            startup_env={},
            lifecycle="per_sample",
            metadata={"sample_id": "sample-1"},
        )
    )

    assert client.create_kwargs is not None
    assert client.create_kwargs["template"] == "gage-swebench-pro-wrapper"
    assert environment.capabilities.supports_upload_download is True

    source = tmp_path / "payload.txt"
    target = tmp_path / "downloaded.txt"
    source.write_text("wrapper file transfer\n", encoding="utf-8")
    asyncio.run(environment.upload_file(source, "/workspace/payload.txt"))
    asyncio.run(environment.download_file("/workspace/payload.txt", target))

    assert target.read_text(encoding="utf-8") == "wrapper file transfer\n"


def test_agentkit_v2_materialization_accepts_swebench_e2b_provider_profile() -> None:
    payload = _swebench_e2b_payload()

    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(payload, source_path=None)
    binding = resolve_agentkit_v2_runtime_binding_specs(materialized, runtime_config=runtime_config)["swe_dut"]

    assert materialized["environments"][0]["provider"] == "e2b"
    assert runtime_config["environments"][0]["provider_config"]["template_id"] == "gage-swebench-pro-wrapper"
    assert binding.environment_provider == "e2b"
    assert binding.environment_profile["profile_id"] == "swebench-e2b-wrapper"
    assert binding.provider_config["template_id"] == "gage-swebench-pro-wrapper"


def test_swebench_e2b_runtime_binding_uses_e2b_profile_for_fresh_verifier() -> None:
    payload = _swebench_e2b_payload()
    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(payload, source_path=None)

    bindings = build_agentkit_v2_runtime_bindings(
        materialized,
        runtime_config=runtime_config,
        backends={"swe_model": object()},
    )

    plan = bindings["swe_dut"].executor_ref.compiled_plan
    assert plan.environment_provider == "e2b"
    assert plan.resource_plan["resource_kind"] == "e2b"
    assert plan.resource_plan["environment_profile"]["provider"] == "e2b"
    assert plan.resource_plan["provider_config"]["template_id"] == "gage-swebench-pro-wrapper"
    assert plan.verifier_environment_policy == "fresh_from_profile"
    assert plan.verifier_environment_profile_id == "swebench-e2b-wrapper"


@pytest.mark.live
def test_live_e2b_wrapper_starts_and_transfers_files(tmp_path: Path) -> None:
    if not os.getenv("E2B_API_KEY"):
        pytest.skip("E2B_API_KEY is required for live E2B wrapper smoke")
    payload = _swebench_e2b_payload()
    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(payload, source_path=None)
    binding_spec = resolve_agentkit_v2_runtime_binding_specs(
        materialized,
        runtime_config=runtime_config,
    )["swe_dut"]
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
                "resource_kind": binding_spec.environment_provider,
                "environment_profile": binding_spec.environment_profile,
                "provider_config": binding_spec.provider_config,
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


def test_docker_and_e2b_verifier_scoring_payload_field_sets_match(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "run_scripts"
    _write_scripts(scripts_dir)

    docker_payload = _run_adapter_payload(provider="docker", scripts_dir=scripts_dir)
    e2b_payload = _run_adapter_payload(provider="e2b", scripts_dir=scripts_dir)

    assert set(_normalize_scoring_payload(docker_payload)) == set(_normalize_scoring_payload(e2b_payload))
    assert set(docker_payload["metric"]) == set(e2b_payload["metric"])
    assert docker_payload["metric"]["score"] == e2b_payload["metric"]["score"] == 1.0


def _swebench_e2b_payload() -> dict[str, Any]:
    return {
        "kind": "AgentEvalConfig",
        "metadata": {"name": "swebench-e2b"},
        "backends": [
            {
                "backend_id": "swe_model",
                "type": "litellm",
                "config": {"model": "gpt-4.1"},
            }
        ],
        "agents": [
            {
                "agent_id": "swe_agent",
                "scheduler": {"type": "framework_loop", "backend_id": "swe_model"},
                "config": {},
            }
        ],
        "benchmarks": [
            {"benchmark_id": "swebench_smoke", "kit_id": "swebench", "config": {"split": "test"}},
        ],
        "environments": [
            {
                "env_id": "swe_e2b",
                "provider": "e2b",
                "profile_id": "swebench-e2b-wrapper",
                "profile": {
                    "asset_dir": "src/gage_eval/agent_eval_kits/swebench/environment/e2b",
                    "capabilities": {"supports_upload_download": True},
                },
                "provider_config": {"template_id": "gage-swebench-pro-wrapper"},
            }
        ],
        "dut_agents": [
            {
                "dut_id": "swe_dut",
                "agent_id": "swe_agent",
                "env_id": "swe_e2b",
                "benchmark_id": "swebench_smoke",
            }
        ],
    }


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
