from __future__ import annotations

import json

import pytest

from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
import gage_eval.role.model.backends.dummy_backend  # noqa: F401
from gage_eval.role.adapters.context_provider import ContextProviderAdapter
from gage_eval.role.adapters.dut_model import DUTModelAdapter
from gage_eval.role.adapters.judge_extend import JudgeExtendAdapter
from gage_eval.role.judge.swebench_docker import SwebenchDocker
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager
from gage_eval.sandbox.base import ExecResult
from gage_eval.sandbox.manager import SandboxManager


class FallbackSandbox:
    def __init__(self, runtime_configs=None, resources=None) -> None:
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}

    def start(self, config):
        return {"container_id": "fallback"}

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        if "bash /workspace/entryscript.sh" in command:
            raise RuntimeError("exec failed")
        if "find . -maxdepth" in command:
            return ExecResult(exit_code=0, stdout="./tests\n./tests/test_bar.py\n", stderr="")
        if "find . -type f" in command:
            return ExecResult(exit_code=0, stdout="./tests/test_bar.py\n", stderr="")
        if "sed -n" in command:
            return ExecResult(exit_code=0, stdout="def test_bar():\n    assert True", stderr="")
        return ExecResult(exit_code=0, stdout="", stderr="")

    def read_file(self, path: str) -> bytes:
        return b""

    def write_file(self, path: str, content: bytes) -> None:
        return None

    def teardown(self) -> None:
        return None


@pytest.mark.io
def test_swebench_fallback_to_docker(tmp_path, mock_trace, temp_workspace, monkeypatch) -> None:
    instance_id = "instance_fallback"
    scripts_root = tmp_path / "run_scripts"
    run_dir = scripts_root / instance_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('ok')\n", encoding="utf-8")

    calls: list[dict] = []

    def fake_run_container(self, image_uri, workspace_dir, params, run_id, instance_id):
        calls.append({"image_uri": image_uri, "workspace_dir": workspace_dir})
        output = {"tests": [{"name": "tests/test_bar.py::test_bar", "status": "PASSED"}]}
        (workspace_dir / "output.json").write_text(json.dumps(output), encoding="utf-8")
        return {"status": "ok"}

    monkeypatch.setattr(SwebenchDocker, "_run_container", fake_run_container)

    samples = [
        {
            "id": instance_id,
            "messages": [{"role": "user", "content": "Fix bug"}],
            "metadata": {
                "instance_id": instance_id,
                "repo": "repo/name",
                "base_commit": "abc",
                "fail_to_pass": ["tests/test_bar.py::test_bar"],
                "pass_to_pass": [],
                "image_uri": "fallback-image:1",
            },
            "sandbox": {
                "sandbox_id": "swebench_runtime",
                "runtime": "fake",
                "lifecycle": "per_sample",
                "image": "fake-image:ignored",
            },
        }
    ]

    planner = TaskPlanner()
    planner.configure_custom_steps(
        [
            {"step": "support", "adapter_id": "swebench_context_provider"},
            {"step": "inference", "adapter_id": "swebench_dut_model"},
            {"step": "judge", "adapter_id": "swebench_docker_judge"},
        ]
    )

    context_adapter = ContextProviderAdapter(
        adapter_id="swebench_context_provider",
        implementation="swebench_repo",
        implementation_params={"repo_source": "docker_image", "repo_root": "/app", "topk_files": 1},
        sandbox_config={"sandbox_id": "swebench_runtime", "lifecycle": "per_sample"},
    )
    model_adapter = DUTModelAdapter(
        adapter_id="swebench_dut_model",
        role_type="dut_model",
        backend={"type": "dummy", "config": {"responses": ["patch"]}},
        capabilities=(),
    )
    judge_adapter = JudgeExtendAdapter(
        adapter_id="swebench_docker_judge",
        implementation="swebench_docker",
        implementation_params={"scripts_dir": str(scripts_root)},
        sandbox_config={"sandbox_id": "swebench_runtime", "lifecycle": "per_sample"},
    )

    resource_profile = ResourceProfile(nodes=[NodeResource(node_id="local", gpus=0, cpus=2)])
    role_manager = RoleManager(resource_profile, concurrency_hint=1)
    role_manager.register_role_adapter("swebench_context_provider", context_adapter)
    role_manager.register_role_adapter("swebench_dut_model", model_adapter)
    role_manager.register_role_adapter("swebench_docker_judge", judge_adapter)

    sandbox_manager = SandboxManager(profiles={"swebench_runtime": {"sandbox_id": "swebench_runtime", "runtime": "fake"}})
    sandbox_manager.register_runtime("fake", FallbackSandbox)

    sample_loop = SampleLoop(samples, concurrency=1, sandbox_manager=sandbox_manager)
    sample_loop.run(planner=planner, role_manager=role_manager, trace=mock_trace)

    assert calls
    assert calls[0]["image_uri"] == "fallback-image:1"
    assert samples[0]["eval_result"]["resolved"] is True
