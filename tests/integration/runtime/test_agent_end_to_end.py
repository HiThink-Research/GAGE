import pytest

from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.adapters.dut_agent import DUTAgentAdapter
from gage_eval.role.toolchain import ToolchainAdapter
from gage_eval.role.agent.backends.base import AgentBackend
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager
from gage_eval.sandbox.base import ExecResult


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
                        "function": {"name": "run_shell", "arguments": "{\"command\": \"echo ok\"}"},
                    }
                ]
            }
        return {"answer": "done"}


@pytest.mark.fast
def test_agent_end_to_end_tool_trace():
    sample = {
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
    }
    samples = [sample]
    planner = TaskPlanner()
    planner.configure_custom_steps(
        [
            {"step": "support", "adapter_id": "toolchain_main"},
            {"step": "inference", "adapter_id": "dut_agent_main"},
        ]
    )
    def command_runner(command, timeout):
        return ExecResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_ms=1.0,
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
                "runtime": "docker",
                "runtime_configs": {"command_runner": command_runner},
            }
        },
        sandbox_config={"sandbox_id": "demo", "lifecycle": "per_sample"},
        max_turns=3,
    )
    resource_profile = ResourceProfile(nodes=[NodeResource(node_id="local", gpus=1, cpus=1)])
    role_manager = RoleManager(resource_profile, concurrency_hint=1)
    role_manager.register_role_adapter("toolchain_main", toolchain)
    role_manager.register_role_adapter("dut_agent_main", dut_agent)
    sample_loop = SampleLoop(
        samples,
        concurrency=1,
        sandbox_profiles={
            "demo": {
                "sandbox_id": "demo",
                "runtime": "docker",
                "runtime_configs": {"command_runner": command_runner},
            }
        },
    )
    trace = ObservabilityTrace()
    sample_loop.run(planner=planner, role_manager=role_manager, trace=trace)

    assert sample["predict_result"]
    agent_trace = sample["predict_result"][0]["agent_trace"]
    assert len(agent_trace) >= 2
    assert agent_trace[0]["trace_role"] == "tool"
    assert agent_trace[0]["output"]["stdout"] == "ok"
