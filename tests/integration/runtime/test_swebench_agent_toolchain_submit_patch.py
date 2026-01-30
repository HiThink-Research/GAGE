from __future__ import annotations

import pytest

from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.adapters.dut_agent import DUTAgentAdapter
from gage_eval.role.agent.backends.base import AgentBackend
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager
from gage_eval.role.toolchain import ToolchainAdapter
from gage_eval.sandbox.base import ExecResult
from gage_eval.sandbox.manager import SandboxManager


class FakeAgentBackend(AgentBackend):
    def __init__(self) -> None:
        self._calls = 0

    def invoke(self, payload):
        self._calls += 1
        if self._calls == 1:
            return {
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "function": {"name": "submit_patch_tool", "arguments": {"timeout_s": 5}},
                    }
                ]
            }
        return {"answer": ""}


class FakeToolSandbox:
    instances: list["FakeToolSandbox"] = []

    def __init__(self, runtime_configs=None, resources=None) -> None:
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}
        self.exec_calls: list[tuple[str, int]] = []
        self.writes: dict[str, bytes] = {}
        FakeToolSandbox.instances.append(self)

    def start(self, config):
        return {"container_id": "fake-tool"}

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        self.exec_calls.append((command, timeout))
        stdout = "diff --git a/foo b/foo\n" if command == "git diff" else ""
        return ExecResult(exit_code=0, stdout=stdout, stderr="", duration_ms=1.0)

    def write_file(self, path: str, content: bytes) -> None:
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.writes[path] = content

    def teardown(self) -> None:
        return None


@pytest.mark.fast
def test_swebench_agent_submit_patch_flow() -> None:
    sample = {
        "id": "s1",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    samples = [sample]
    planner = TaskPlanner()
    planner.configure_custom_steps(
        [
            {"step": "support", "adapter_id": "toolchain_main"},
            {"step": "inference", "adapter_id": "dut_agent_main"},
        ]
    )

    toolchain = ToolchainAdapter(
        adapter_id="toolchain_main",
        tools=[
            {
                "name": "submit_patch_tool",
                "parameters": {"type": "object", "properties": {"timeout_s": {"type": "integer"}}},
            }
        ],
    )
    sandbox_profiles = {"toolbox": {"sandbox_id": "toolbox", "runtime": "fake"}}
    dut_agent = DUTAgentAdapter(
        adapter_id="dut_agent_main",
        role_type="dut_agent",
        capabilities=(),
        agent_backend=FakeAgentBackend(),
        sandbox_profiles=sandbox_profiles,
        sandbox_config={"sandbox_id": "toolbox", "lifecycle": "per_sample"},
        max_turns=3,
    )
    resource_profile = ResourceProfile(nodes=[NodeResource(node_id="local", gpus=0, cpus=1)])
    role_manager = RoleManager(resource_profile, concurrency_hint=1)
    role_manager.register_role_adapter("toolchain_main", toolchain)
    role_manager.register_role_adapter("dut_agent_main", dut_agent)

    sandbox_manager = SandboxManager(profiles=sandbox_profiles)
    sandbox_manager.register_runtime("fake", FakeToolSandbox)

    sample_loop = SampleLoop(samples, concurrency=1, sandbox_manager=sandbox_manager)
    trace = ObservabilityTrace()
    sample_loop.run(planner=planner, role_manager=role_manager, trace=trace)

    sandbox = FakeToolSandbox.instances[-1]
    assert ("git diff", 5) in sandbox.exec_calls
    assert "/workspace/submission.patch" in sandbox.writes
