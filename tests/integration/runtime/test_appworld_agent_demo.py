from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config import build_default_registry
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from tests.integration.runtime.mcp_stub import AppWorldMcpStub


class DemoAgent:
    """Stateful agent stub that emits a tool call then an answer."""

    def __init__(self) -> None:
        self._calls = 0

    def run(self, payload: dict) -> dict:
        self._calls += 1
        if self._calls == 1:
            return {
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "function": {
                            "name": "step",
                            "arguments": {"action": "noop"},
                        },
                    }
                ]
            }
        return {"answer": "done"}


@pytest.mark.io
def test_appworld_agent_demo_end_to_end(
    temp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "appworld_agent_demo.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    stub = AppWorldMcpStub()
    def requester(method: str, payload: dict) -> dict:
        return stub.requester(method, payload)
    payload["agent_backends"][0]["type"] = "agent_class"
    payload["agent_backends"][0]["config"] = {"agent_class": DemoAgent, "method": "run"}
    payload["backends"][0]["type"] = "dummy"
    payload["backends"][0]["config"] = {"responses": ["ok"]}
    payload["mcp_clients"][0]["endpoint"] = "http://stub"
    payload["mcp_clients"][0]["params"] = {**payload["mcp_clients"][0].get("params", {}), "requester": requester}
    payload["tasks"][0]["max_samples"] = 1
    dut_agent = next(item for item in payload["role_adapters"] if item.get("adapter_id") == "dut_agent_main")
    dut_agent.setdefault("params", {})
    dut_agent["params"]["pre_hooks"] = []
    dut_agent["params"]["post_hooks"] = []
    payload["sandbox_profiles"][0]["runtime_configs"]["start_container"] = False

    config = PipelineConfig.from_dict(payload)
    registry = build_default_registry()
    resource_profile = ResourceProfile(nodes=[NodeResource(node_id="local", gpus=0, cpus=1)])
    trace = ObservabilityTrace()

    runtime = build_runtime(config, registry, resource_profile, trace=trace)
    captured: list[dict] = []
    for entry in runtime._tasks:
        entry.sample_loop.register_hook(lambda sample, store=captured: store.append(sample))

    runtime.run()

    assert captured
    sample = captured[0]
    predict = sample.get("predict_result")
    assert predict
    agent_trace = predict[0].get("agent_trace")
    assert agent_trace
    tool_steps = [step for step in agent_trace if step.get("trace_role") == "tool"]
    assert tool_steps
    assert tool_steps[0]["name"] == "step"
    assert tool_steps[0]["output"]["observation"]["status"] == "ok"
    assert stub.tool_calls and stub.tool_calls[0]["name"] == "step"
