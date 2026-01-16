from __future__ import annotations

import pytest

from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.role.adapters.dut_agent import DUTAgentAdapter
from gage_eval.role.adapters.toolchain import ToolchainAdapter
from gage_eval.role.agent.backends.base import AgentBackend
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager
from gage_eval.sandbox.base import ExecResult
from gage_eval.sandbox.manager import SandboxManager


class FakeAgentBackend(AgentBackend):
    def invoke(self, payload):
        if payload.get("turn_index") == 1:
            return {
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "function": {"name": "run_shell", "arguments": "{\"command\": \"echo ok\"}"},
                    }
                ]
            }
        return {"answer": "done"}


class FakeSandbox:
    start_calls = 0
    teardown_calls = 0

    def __init__(self, runtime_configs=None, resources=None):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}

    def start(self, config):
        FakeSandbox.start_calls += 1
        return {"container_id": f"fake-{FakeSandbox.start_calls}"}

    def exec(self, command, timeout=30):
        return ExecResult(exit_code=0, stdout="ok", stderr="", duration_ms=1.0)

    def teardown(self):
        FakeSandbox.teardown_calls += 1


@pytest.mark.fast
def test_appworld_per_sample_sandbox_lifecycle(mock_trace) -> None:
    FakeSandbox.start_calls = 0
    FakeSandbox.teardown_calls = 0

    samples = [
        {
            "id": "s1",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_shell",
                        "description": "Run shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ],
        },
        {
            "id": "s2",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_shell",
                        "description": "Run shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ],
        },
    ]
    planner = TaskPlanner()
    planner.configure_custom_steps(
        [
            {"step": "support", "adapter_id": "toolchain_main"},
            {"step": "inference", "adapter_id": "dut_agent_main"},
        ]
    )
    toolchain = ToolchainAdapter(adapter_id="toolchain_main", tools=[])
    dut_agent = DUTAgentAdapter(
        adapter_id="dut_agent_main",
        role_type="dut_agent",
        capabilities=(),
        agent_backend=FakeAgentBackend(),
        sandbox_profiles={
            "demo": {
                "sandbox_id": "demo",
                "runtime": "fake",
            }
        },
        sandbox_config={"sandbox_id": "demo", "lifecycle": "per_sample"},
        max_turns=3,
    )
    resource_profile = ResourceProfile(nodes=[NodeResource(node_id="local", gpus=0, cpus=1)])
    role_manager = RoleManager(resource_profile, concurrency_hint=1)
    role_manager.register_role_adapter("toolchain_main", toolchain)
    role_manager.register_role_adapter("dut_agent_main", dut_agent)
    sandbox_manager = SandboxManager(profiles={"demo": {"sandbox_id": "demo", "runtime": "fake"}})
    sandbox_manager.register_runtime("fake", FakeSandbox)
    sample_loop = SampleLoop(
        samples,
        concurrency=1,
        sandbox_manager=sandbox_manager,
    )

    sample_loop.run(planner=planner, role_manager=role_manager, trace=mock_trace)

    assert FakeSandbox.start_calls == 2
    assert FakeSandbox.teardown_calls == 2
    assert all(sample.get("predict_result") for sample in samples)
    for sample in samples:
        agent_trace = sample["predict_result"][0]["agent_trace"]
        tool_steps = [step for step in agent_trace if step.get("trace_role") == "tool"]
        assert tool_steps

    release_events = [e for e in mock_trace.events if e["event"] == "sandbox_release_end"]
    assert len(release_events) == 2
