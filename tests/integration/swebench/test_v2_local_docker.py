from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from gage_eval.agent_eval_kits.swebench.judge.adapters import SwebenchVerifierAdapter
from gage_eval.agent_eval_kits.swebench.kit import load_kit
from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.verifier.contracts import VerifierInput
from gage_eval.environment import EnvironmentResources
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.providers.docker import DockerEnvironmentConfig, DockerEnvironmentProvider


DOCKER_SMOKE_IMAGE = "python:3.12-bookworm"
DOCKER_SMOKE_PLATFORM = "linux/amd64"


def _docker_client_or_skip() -> Any:
    docker = pytest.importorskip("docker", reason="docker SDK is not installed")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        pytest.skip(f"Docker daemon is not available: {exc}")
    return client


def _ensure_smoke_image_or_skip(client: Any) -> None:
    try:
        client.images.get(DOCKER_SMOKE_IMAGE)
        return
    except Exception:
        pass
    try:
        client.images.pull(DOCKER_SMOKE_IMAGE, platform=DOCKER_SMOKE_PLATFORM)
    except Exception as exc:
        pytest.skip(f"Docker smoke image is not available and pull failed: {exc}")


async def _create_docker_environment() -> Any:
    provider = DockerEnvironmentProvider()
    return await provider.create(
        kit_id="swebench",
        provider="docker",
        profile_id="swebench_runtime",
        profile=EnvironmentProfile(profile_id="swebench_runtime", provider="docker", config={}),
        provider_config=DockerEnvironmentConfig(
            image=DOCKER_SMOKE_IMAGE,
            docker_platform=DOCKER_SMOKE_PLATFORM,
            workdir="/workspace",
            network_policy="block",
        ),
        resources=EnvironmentResources(cpu=1.0, memory_gb=1.0, network_policy="block"),
        startup_env={},
        lifecycle="per_sample",
        metadata={"sample_id": "swebench-docker-smoke"},
    )


def test_trials_two_smoke_writes_two_trial_artifacts(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))

    for trial_id in ("trial_0001", "trial_0002"):
        sink.write_artifact(
            run_id="run-smoke",
            task_id="task-1",
            sample_id="sample-1",
            trial_id=trial_id,
            owner="verifier",
            name="verifier_result.json",
            content={"trial_id": trial_id, "resolved": trial_id == "trial_0002"},
            mime_type="application/json",
        )

    for trial_id in ("trial_0001", "trial_0002"):
        result_path = (
            tmp_path
            / "run-smoke"
            / "artifacts"
            / "task-1"
            / "sample-1"
            / "trials"
            / trial_id
            / "verifier"
            / "verifier_result.json"
        )
        assert json.loads(result_path.read_text(encoding="utf-8"))["trial_id"] == trial_id


@pytest.mark.io
def test_swebench_v2_local_docker_verifier_smoke(tmp_path: Path) -> None:
    client = _docker_client_or_skip()
    _ensure_smoke_image_or_skip(client)
    kit = load_kit()
    assert kit.default_environment_provider == "docker"
    assert kit.verifier_environment_profile_id == "swebench_runtime"

    environment = asyncio.run(_create_docker_environment())
    try:
        probe = asyncio.run(environment.exec("command -v git && command -v python", timeout_s=20))
        if probe.exit_code != 0:
            pytest.skip("Docker smoke image lacks git or python")
        setup = asyncio.run(
            environment.exec(
                "mkdir -p /app && cd /app && "
                "git init && git config user.email smoke@example.com && "
                "git config user.name smoke && "
                "printf 'old\\n' > foo.txt && git add foo.txt && git commit -m init >/dev/null && "
                "git rev-parse HEAD",
                timeout_s=30,
            )
        )
        if setup.exit_code != 0:
            pytest.skip(f"Docker smoke could not initialize repo: {setup.stderr}")
        base_commit = setup.stdout.strip().splitlines()[-1]

        scripts_dir = tmp_path / "run_scripts" / "instance_1"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
        (scripts_dir / "parser.py").write_text(
            "import json, sys\n"
            "json.dump({'tests': [{'name': 'tests/test_fix.py::test_fix', 'status': 'PASSED'}]}, open(sys.argv[3], 'w'))\n",
            encoding="utf-8",
        )
        patch = (
            "diff --git a/foo.txt b/foo.txt\n"
            "index 3367afd..3e75765 100644\n"
            "--- a/foo.txt\n"
            "+++ b/foo.txt\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )

        result = SwebenchVerifierAdapter(scripts_dir=str(tmp_path / "run_scripts")).run(
            VerifierInput(
                benchmark_kit_id="swebench",
                scheduler_type="framework_loop",
                sample_id="instance_1",
                sample={
                    "id": "instance_1",
                    "metadata": {
                        "instance_id": "instance_1",
                        "repo": "repo/name",
                        "base_commit": base_commit,
                        "fail_to_pass": ["tests/test_fix.py::test_fix"],
                        "pass_to_pass": [],
                    },
                },
                scheduler_result={"agent_output": {"patch_content": patch}},
                runtime_context={"environment": environment},
                verifier_resources={"test_timeout_s": 60},
            )
        )

        assert result.payload["resolved"] is True
    finally:
        asyncio.run(environment.stop(delete=True))


@pytest.mark.io
def test_swebench_v2_local_docker_applies_test_patch_after_submission_patch(tmp_path: Path) -> None:
    client = _docker_client_or_skip()
    _ensure_smoke_image_or_skip(client)
    environment = asyncio.run(_create_docker_environment())
    try:
        setup = asyncio.run(
            environment.exec(
                "mkdir -p /app && cd /app && "
                "git init && git config user.email smoke@example.com && "
                "git config user.name smoke && "
                "printf 'old\\n' > foo.txt && printf 'base\\n' > bar.txt && "
                "git add foo.txt bar.txt && git commit -m init >/dev/null && "
                "git rev-parse HEAD",
                timeout_s=30,
            )
        )
        if setup.exit_code != 0:
            pytest.skip(f"Docker smoke could not initialize repo: {setup.stderr}")
        base_commit = setup.stdout.strip().splitlines()[-1]

        scripts_dir = tmp_path / "run_scripts" / "instance_1"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "run_script.sh").write_text(
            "#!/bin/bash\n"
            "cd /app\n"
            "grep -qx new foo.txt && grep -qx harness bar.txt && echo ok\n",
            encoding="utf-8",
        )
        (scripts_dir / "parser.py").write_text(
            "import json, sys\n"
            "stdout = open(sys.argv[1]).read()\n"
            "status = 'PASSED' if 'ok' in stdout else 'FAILED'\n"
            "json.dump({'tests': [{'name': 'tests/test_fix.py::test_fix', 'status': status}]}, open(sys.argv[3], 'w'))\n",
            encoding="utf-8",
        )
        patch = (
            "diff --git a/foo.txt b/foo.txt\n"
            "--- a/foo.txt\n"
            "+++ b/foo.txt\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        test_patch = (
            "diff --git a/bar.txt b/bar.txt\n"
            "--- a/bar.txt\n"
            "+++ b/bar.txt\n"
            "@@ -1 +1 @@\n"
            "-base\n"
            "+harness\n"
        )

        result = SwebenchVerifierAdapter(scripts_dir=str(tmp_path / "run_scripts")).run(
            VerifierInput(
                benchmark_kit_id="swebench",
                scheduler_type="framework_loop",
                sample_id="instance_1",
                sample={
                    "id": "instance_1",
                    "metadata": {
                        "instance_id": "instance_1",
                        "repo": "repo/name",
                        "base_commit": base_commit,
                        "fail_to_pass": ["tests/test_fix.py::test_fix"],
                        "pass_to_pass": [],
                        "test_patch": test_patch,
                    },
                },
                scheduler_result={"agent_output": {"patch_content": patch}},
                runtime_context={"environment": environment},
                verifier_resources={"test_timeout_s": 60},
            )
        )

        assert result.payload["resolved"] is True
    finally:
        asyncio.run(environment.stop(delete=True))


@pytest.mark.live
@pytest.mark.io
def test_real_nodebb_instance_resolves_with_gold_patch() -> None:
    if os.getenv("GAGE_EVAL_RUN_REAL_SWEBENCH_PRO") != "1":
        pytest.skip("Set GAGE_EVAL_RUN_REAL_SWEBENCH_PRO=1 to run live SWE-bench Pro verifier")
    image = os.getenv("SWEBENCH_PRO_REAL_IMAGE")
    base_commit = os.getenv("SWEBENCH_PRO_REAL_BASE_COMMIT")
    gold_patch = os.getenv("SWEBENCH_PRO_REAL_GOLD_PATCH")
    fail_to_pass = [item for item in os.getenv("SWEBENCH_PRO_REAL_FAIL_TO_PASS", "").split(",") if item]
    if not image or not base_commit or not gold_patch or not fail_to_pass:
        pytest.skip(
            "SWEBENCH_PRO_REAL_IMAGE, SWEBENCH_PRO_REAL_BASE_COMMIT, "
            "SWEBENCH_PRO_REAL_GOLD_PATCH, and SWEBENCH_PRO_REAL_FAIL_TO_PASS are required"
        )
    scripts_dir = Path("third_party/swebench_pro/run_scripts")
    instance_id = "instance_NodeBB__NodeBB-f1a80d48cc45877fcbadf34c2345dd9709722c7f-v4fbcfae8b15e4ce5d132c408bca69ebb9cf146ed"
    if not (scripts_dir / instance_id / "run_script.sh").exists():
        pytest.skip(f"Vendored SWE-bench Pro run script missing: {instance_id}")

    provider = DockerEnvironmentProvider()
    environment = asyncio.run(
        provider.create(
            kit_id="swebench",
            provider="docker",
            profile_id="swebench_runtime",
            profile=EnvironmentProfile(profile_id="swebench_runtime", provider="docker", config={}),
            provider_config=DockerEnvironmentConfig(
                image=image,
                docker_platform=DOCKER_SMOKE_PLATFORM,
                workdir="/workspace",
                network_policy="block",
            ),
            resources=EnvironmentResources(cpu=2.0, memory_gb=4.0, network_policy="block"),
            startup_env={},
            lifecycle="per_sample",
            metadata={"sample_id": instance_id},
        )
    )
    try:
        result = SwebenchVerifierAdapter(scripts_dir=str(scripts_dir)).run(
            VerifierInput(
                benchmark_kit_id="swebench",
                scheduler_type="framework_loop",
                sample_id=instance_id,
                sample={
                    "id": instance_id,
                    "metadata": {
                        "instance_id": instance_id,
                        "repo": "NodeBB/NodeBB",
                        "base_commit": base_commit,
                        "fail_to_pass": fail_to_pass,
                        "pass_to_pass": [],
                    },
                },
                scheduler_result={"agent_output": {"patch_content": gold_patch}},
                runtime_context={"environment": environment},
                verifier_resources={"test_timeout_s": int(os.getenv("SWEBENCH_PRO_REAL_TIMEOUT_S", "900"))},
            )
        )

        assert result.payload["resolved"] is True
    finally:
        asyncio.run(environment.stop(delete=True))
