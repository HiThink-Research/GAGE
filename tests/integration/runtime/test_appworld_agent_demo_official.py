from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import gage_eval.agent_eval_kits.appworld.runtime as appworld_runtime_module
from gage_eval.config import build_default_registry
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.sandbox.integrations.appworld.mcp_client import AppWorldStreamableMcpClient
from tests._support.stubs.mcp_stub import AppWorldMcpStub


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


class _ResponseStub:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return dict(self._payload)


@pytest.mark.io
def test_appworld_demo_with_streamable_http(
    temp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "appworld"
        / "appworld_agent_demo.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    stub = AppWorldMcpStub()
    lifecycle_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_post(url: str, json: dict[str, Any], timeout: int):
        method = url.rsplit("/", 1)[-1]
        lifecycle_calls.append((method, dict(json or {})))
        return _ResponseStub({"output": {"status": "ok", "method": method, "task_id": json.get("task_id")}})

    def requester(method: str, payload: dict) -> dict:
        return stub.requester(method, payload)

    monkeypatch.setattr(appworld_runtime_module.requests, "post", fake_post)
    payload["backends"][0]["type"] = "dummy"
    payload["backends"][0]["config"] = {"responses": ["ok"]}
    payload["mcp_clients"][0]["endpoint"] = "http://stub"
    payload["mcp_clients"][0]["params"] = {**payload["mcp_clients"][0].get("params", {}), "requester": requester}
    payload["tasks"][0]["max_samples"] = 1
    payload["tasks"][0]["steps"] = [
        {"step": "support", "adapter_id": "toolchain_main"},
        {"step": "inference", "adapter_id": "dut_agent_main"},
    ]
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
    assert [name for name, _ in lifecycle_calls] == ["initialize", "save"]
