from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config import build_default_registry
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.sandbox.integrations.appworld.mcp_client import AppWorldStreamableMcpClient
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
def test_appworld_demo_with_streamable_http(
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
    payload["mcp_clients"][0]["endpoint"] = "http://stub"
    payload["mcp_clients"][0]["params"] = {**payload["mcp_clients"][0].get("params", {}), "requester": requester}
    payload["tasks"][0]["max_samples"] = 1
    dut_agent = next(item for item in payload["role_adapters"] if item.get("adapter_id") == "dut_agent_main")
    dut_agent.setdefault("params", {})
    dut_agent["params"]["pre_hooks"] = []
    dut_agent["params"]["post_hooks"] = []
    toolchain = next(item for item in payload["role_adapters"] if item.get("adapter_id") == "toolchain_main")
    toolchain.setdefault("params", {})
    toolchain["params"]["tool_allowlist"] = ["step", "get_state"]
    toolchain["params"]["tool_prefixes"] = []
    payload["sandbox_profiles"][0]["runtime_configs"]["start_container"] = False

    config = PipelineConfig.from_dict(payload)
    registry = build_default_registry()
    clients = registry.materialize_mcp_clients(config)
    assert isinstance(clients["appworld_env"], AppWorldStreamableMcpClient)

    resource_profile = ResourceProfile(nodes=[NodeResource(node_id="local", gpus=0, cpus=1)])
    trace = ObservabilityTrace()
    runtime = build_runtime(config, registry, resource_profile, trace=trace)
    runtime.run()

    assert stub.tool_calls and stub.tool_calls[0]["name"] == "step"
