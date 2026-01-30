from __future__ import annotations

import json

import pytest

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
import gage_eval.metrics.builtin.swebench  # noqa: F401
from gage_eval.role.adapters.dut_model import DUTModelAdapter
from gage_eval.role.adapters.judge_extend import JudgeExtendAdapter
import gage_eval.role.model.backends.dummy_backend  # noqa: F401
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager
from gage_eval.sandbox.base import ExecResult
from gage_eval.sandbox.manager import SandboxManager


class FallbackSandbox:
    def __init__(self, runtime_configs=None, resources=None) -> None:
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}
        self.files: dict[str, bytes] = {}

    def start(self, config):
        output = {"tests": [{"name": "tests/test_bar.py::test_bar", "status": "PASSED"}]}
        self.files["/workspace/output.json"] = json.dumps(output).encode("utf-8")
        self.files["/workspace/submission.patch"] = (
            "diff --git a/foo.txt b/foo.txt\n"
            "index 1111111..2222222 100644\n"
            "--- a/foo.txt\n"
            "+++ b/foo.txt\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        ).encode("utf-8")
        return {"container_id": "fallback"}

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        if command.startswith("cat /workspace/submission.patch"):
            return ExecResult(exit_code=0, stdout=self.files["/workspace/submission.patch"].decode("utf-8"), stderr="")
        return ExecResult(exit_code=0, stdout="", stderr="")

    def read_file(self, path: str) -> bytes:
        return self.files.get(path, b"")

    def write_file(self, path: str, content: bytes) -> None:
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.files[path] = content

    def teardown(self) -> None:
        return None


@pytest.mark.io
def test_swebench_submission_patch_fallback_flow(tmp_path, mock_trace, temp_workspace) -> None:
    instance_id = "instance_1"
    scripts_root = tmp_path / "run_scripts"
    run_dir = scripts_root / instance_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('ok')\n", encoding="utf-8")

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
            },
            "sandbox": {
                "sandbox_id": "swebench_runtime",
                "runtime": "fake",
                "lifecycle": "per_sample",
                "image": "fake-image:1",
            },
        }
    ]

    planner = TaskPlanner()
    planner.configure_custom_steps(
        [
            {"step": "inference", "adapter_id": "swebench_dut_model"},
            {"step": "judge", "adapter_id": "swebench_docker_judge"},
            {"step": "auto_eval"},
        ]
    )
    planner.configure_metrics([MetricSpec(metric_id="swebench_resolve_rate", implementation="swebench_resolve_rate")])

    model_adapter = DUTModelAdapter(
        adapter_id="swebench_dut_model",
        role_type="dut_model",
        backend={"type": "dummy", "config": {"responses": [""]}},
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
    role_manager.register_role_adapter("swebench_dut_model", model_adapter)
    role_manager.register_role_adapter("swebench_docker_judge", judge_adapter)

    sandbox_manager = SandboxManager(profiles={"swebench_runtime": {"sandbox_id": "swebench_runtime", "runtime": "fake"}})
    sandbox_manager.register_runtime("fake", FallbackSandbox)

    sample_loop = SampleLoop(samples, concurrency=1, sandbox_manager=sandbox_manager)
    sample_loop.run(planner=planner, role_manager=role_manager, trace=mock_trace)

    assert samples[0]["eval_result"]["resolved"] is True
    events = [item["event"] for item in mock_trace.events]
    assert "swebench_patch_fallback" in events
