from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import gage_eval.agent_eval_kits.appworld.runtime as appworld_runtime_module
import gage_eval.agent_eval_kits.appworld.sub_workflows.framework_loop as appworld_framework_loop_module
from gage_eval.config import build_default_registry
from gage_eval.config.loader import load_pipeline_config_payload
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.registry import registry
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.agent_eval_kits.appworld.mcp_client import AppWorldStreamableMcpClient
from tests._support.stubs.mcp_stub import AppWorldMcpStub


class DemoModelBackend:
    """Stateful static model backend that emits a tool call then an answer."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        del config
        self._calls = 0

    async def ainvoke(self, payload: dict) -> dict:
        del payload
        self._calls += 1
        if self._calls == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "tool-1",
                                    "type": "function",
                                    "function": {
                                        "name": "step",
                                        "arguments": "{\"action\":\"noop\"}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}


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
    payload = load_pipeline_config_payload(config_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    stub = AppWorldMcpStub()
    lifecycle_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_post(url: str, json: dict[str, Any], timeout: int):
        method = url.rsplit("/", 1)[-1]
        lifecycle_calls.append((method, dict(json or {})))
        return _ResponseStub({"output": {"status": "ok", "method": method, "task_id": json.get("task_id")}})

    def requester(method: str, payload: dict) -> dict:
        return stub.requester(method, payload)

    def fake_fetch_mcp_tool_schemas(
        mcp_endpoint: str,
        mcp_client_id: str | None,
        *,
        allowed_apps: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        del mcp_endpoint, allowed_apps
        return [_tool_schema(tool, mcp_client_id or "appworld_env") for tool in stub.list_tools()]

    monkeypatch.setattr(appworld_runtime_module.requests, "post", fake_post)
    monkeypatch.setattr(
        appworld_framework_loop_module,
        "fetch_mcp_tool_schemas",
        fake_fetch_mcp_tool_schemas,
    )
    _configure_runtime_owned_appworld_agent(payload, temp_workspace=temp_workspace)
    payload["mcp_clients"][0]["endpoint"] = "http://stub"
    payload["mcp_clients"][0]["params"] = {**payload["mcp_clients"][0].get("params", {}), "requester": requester}
    payload["tasks"][0]["max_samples"] = 1
    payload["tasks"][0]["steps"] = [{"step": "inference", "adapter_id": "dut_agent_main"}]

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


def _configure_runtime_owned_appworld_agent(payload: dict[str, Any], *, temp_workspace: Path) -> None:
    registry.register(
        "backends",
        "appworld_demo_tool_backend_official",
        DemoModelBackend,
        desc="AppWorld streamable MCP integration test backend",
        tags=("test", "agent_runtime"),
    )
    backend_id = payload["backends"][0]["backend_id"]
    payload.pop("agent_backends", None)
    payload["backends"][0]["type"] = "appworld_demo_tool_backend_official"
    payload["backends"][0]["config"] = {}
    dut = next(adapter for adapter in payload["role_adapters"] if adapter["adapter_id"] == "dut_agent_main")
    dut.pop("agent_backend_id", None)
    dut["agent_runtime_id"] = "appworld_framework_loop"
    dut["backend_id"] = backend_id
    params = dut.setdefault("params", {})
    params.pop("pre_hooks", None)
    params.pop("post_hooks", None)
    params["max_turns"] = 3
    _configure_stub_environment(payload, params, temp_workspace=temp_workspace)


def _configure_stub_environment(
    payload: dict[str, Any],
    params: dict[str, Any],
    *,
    temp_workspace: Path,
) -> None:
    metadata = {
        "env_endpoint": "http://stub-env",
        "apis_endpoint": "http://stub-apis",
        "mcp_endpoint": "http://stub",
    }
    provider_config = {"base_cwd": str(temp_workspace)}
    environment_profile = {
        "provider": "local_process",
        "profile_id": "appworld_local_stub",
        "config": provider_config,
        "metadata": metadata,
        "resources": {"cpu": 1, "memory_gb": 1.0},
    }
    params["environment_profile"] = environment_profile
    params["provider_config"] = provider_config
    params["resources"] = {"cpu": 1, "memory_gb": 1.0}
    params["startup_env"] = {}
    params["lifecycle"] = "per_sample"
    if payload.get("environments"):
        payload["environments"][0] = {
            "env_id": "appworld_local_stub",
            "provider": "local_process",
            "profile_id": "appworld_local_stub",
            "profile": {"metadata": metadata},
            "provider_config": provider_config,
            "resources": {"cpu": 1, "memory_gb": 1.0},
            "lifecycle": "per_sample",
        }


def _tool_schema(tool: dict[str, Any], mcp_client_id: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "parameters": dict(tool.get("inputSchema") or {}),
        },
        "x-gage": {"mcp_client_id": mcp_client_id},
    }
