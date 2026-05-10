from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_runtime.contracts.failure import FailureEnvelopeError
from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.agent_runtime.resolver import build_compiled_runtime_executor, compile_agent_runtime_plan
from gage_eval.agent_runtime.schedulers.framework_loop import FrameworkLoopScheduler
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.tooling.contracts import ToolExecutionContext, ToolSchemaIR
from gage_eval.agent_runtime.tooling.mcp.client import McpServerProcess
from gage_eval.agent_runtime.tooling.registry import RuntimeToolRegistry
from gage_eval.agent_runtime.tooling.router import ToolRouter
from gage_eval.agent_runtime.verifier.contracts import RuntimeJudgeOutcome, VerifierInput, VerifierResult
from gage_eval.agent_eval_kits.tau2.tools import build_tool_registry as build_tau2_tool_registry
from gage_eval.environment.contracts import ExecResult
from gage_eval.environment.lease import EnvironmentLease


class _ToolCallingBackend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        if len(self.payloads) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-echo",
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"text":"hi"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}


class _FinalBackend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        return {"answer": "done", "agent_trace": []}


class _InvalidJsonBackend:
    async def ainvoke(self, payload: dict[str, Any]) -> str:
        del payload
        return "<not-json>"


class _TimeoutBackend:
    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        raise TimeoutError("model request timed out")


class _LitellmTimeout(Exception):
    pass


class _LitellmTimeoutBackend:
    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        raise _LitellmTimeout("request timeout from model provider")


class _CostBackend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        return {
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            "usage": {"cost_usd": 6.0},
        }


class _RunShellBackend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        if len(self.payloads) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-shell",
                                    "type": "function",
                                    "function": {
                                        "name": "run_shell",
                                        "arguments": '{"command":"echo from-tool","timeout_s":"5"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}


class _McpToolBackend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        if len(self.payloads) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-mcp",
                                    "type": "function",
                                    "function": {
                                        "name": "mcp_lookup",
                                        "arguments": '{"q":"refund"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}


class _Tau2TextBackend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        if len(self.payloads) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": 'call:respond{message:"hello",urgent:"true",count:"12"}',
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}


class _Tau2PlainTextBackend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        return {"choices": [{"message": {"content": "plain text without required tool"}, "finish_reason": "stop"}]}


class _Tau2EmptyTextBackend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        return {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}


class _SubmitReviewThenProseBackend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        if len(self.payloads) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-submit",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_patch_tool",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return {"choices": [{"message": {"content": "final prose only"}, "finish_reason": "stop"}]}


class _Tau2ConversationBackend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        if len(self.payloads) == 1:
            return {"choices": [{"message": {"content": "Could you tell me which line?"}, "finish_reason": "stop"}]}
        return {"choices": [{"message": {"content": "###STOP###"}, "finish_reason": "stop"}]}


class _Tau2EnvironmentLease:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def exec_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        return {"final_answer": arguments.get("message"), "user_message": arguments.get("message")}


class _Tau2NonTerminalEnvironmentLease:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def exec_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        return {"user_message": "continue"}


class _Tau2ScriptedRespondLease:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def exec_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        if len(self.calls) == 1:
            return {"user_message": "The line ending in 1234."}
        return {"final_answer": "###STOP###", "user_message": "###STOP###"}


class _RawProviderSchema:
    def __init__(self) -> None:
        self.name = "raw-provider-schema"


class _McpClient:
    def __init__(self, tool_name: str = "mcp_tool") -> None:
        self.tool_name = tool_name
        self.calls: list[tuple[str, Any]] = []
        self.reset_count = 0

    def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": self.tool_name, "inputSchema": {"type": "object"}}]

    async def call_tool(self, name: str, arguments: Any) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"mcp": name, "arguments": arguments}

    def reset(self) -> None:
        self.reset_count += 1


class _FakeEnvironment:
    env_id = "env-1"
    name = "fake-runtime"
    provider = "docker"
    capabilities = {}
    metadata = {"runtime_handle": {"environment_endpoint": "fake://environment"}}

    def __init__(self) -> None:
        self.commands: list[tuple[str, int]] = []
        self.files: dict[str, bytes] = {}
        self.teardown_called = False

    async def exec(self, command: str, **kwargs: Any) -> ExecResult:
        timeout = int(kwargs.get("timeout_s") or 30)
        self.commands.append((command, timeout))
        if "find . -type f" in command:
            return ExecResult(command=command, exit_code=0, stdout="", stderr="")
        if command == "echo from-tool":
            return ExecResult(command=command, exit_code=0, stdout="from env\n", stderr="")
        return ExecResult(command=command, exit_code=0, stdout="", stderr="")

    async def read_file(self, path: str, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
        del max_bytes
        return self.files.get(path, b"")

    async def write_file(self, path: str, content: bytes | str) -> None:
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.files[path] = bytes(content)

    async def stop(self, *, delete: bool = True) -> None:
        del delete
        self.teardown_called = True


class _FakeEnvironmentManager:
    def __init__(self) -> None:
        self.environment = _FakeEnvironment()
        self.released = False

    async def acquire(self, **kwargs: Any) -> EnvironmentLease:
        return EnvironmentLease(
            lease_id="env-lease-1",
            environment=self.environment,
            provider=str(kwargs["provider"]),
            profile_id=str(kwargs["profile_id"]),
            lifecycle=kwargs["lifecycle"],
            exclusive=True,
            metadata=dict(kwargs["metadata"]),
        )

    async def release(self, lease: EnvironmentLease) -> None:
        self.released = True
        await lease.environment.stop(delete=True)


class _PassingVerifierRunner:
    def run(
        self,
        *,
        plan,
        session,
        sample,
        scheduler_result,
        sandbox_provider=None,
        environment_lease=None,
    ) -> RuntimeJudgeOutcome:
        del plan, sandbox_provider, environment_lease
        payload = {"status": "completed", "resolved": True, "score": 1.0}
        verifier_input = VerifierInput(
            benchmark_kit_id=session.benchmark_kit_id,
            scheduler_type=session.scheduler_type,
            sample_id=session.sample_id,
            sample=sample,
            scheduler_result=scheduler_result.to_dict(),
        )
        return RuntimeJudgeOutcome(
            verifier_input=verifier_input,
            verifier_result=VerifierResult(status="completed", payload=payload),
            judge_output=payload,
            persisted_path=session.artifact_layout["verifier_result"],
        )

    def build_failed_outcome(self, *, plan, session, sample, failure):
        raise AssertionError(f"unexpected runtime failure: {failure}")


def test_framework_loop_uses_tool_ir_router_and_injects_provider_tool_result() -> None:
    dispatches: list[tuple[dict[str, Any], ToolExecutionContext]] = []
    registry = RuntimeToolRegistry()
    registry.register_local_function(
        ToolSchemaIR(
            name="echo",
            description="Echo text",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            raw_schema={"name": "echo"},
        ),
        lambda arguments, context: _record_dispatch(arguments, context, dispatches),
    )
    backend = _ToolCallingBackend()
    scheduler = FrameworkLoopScheduler(
        backend=backend,
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        max_turns=2,
    )

    result = asyncio.run(
        scheduler.arun(
            session=_session(),
            sample={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "raw_sample_tool", "inputSchema": {"type": "object"}}],
            },
            payload={},
            workflow_bundle=_bundle(
                inject_tool_schemas=lambda **_: (_ for _ in ()).throw(AssertionError("raw tool injection used"))
            ),
            sandbox_provider=None,
        )
    )

    assert result.status == "completed"
    assert result.agent_output["answer"] == "done"
    assert backend.payloads[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo text",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            },
        }
    ]
    assert dispatches[0][0] == {"text": "hi"}
    assert dispatches[0][1].run_id == "run-1"
    assert dispatches[0][1].task_id == "task-1"
    assert dispatches[0][1].sample_id == "sample-1"
    assert dispatches[0][1].trial_id == "trial_0001"
    assert dispatches[0][1].resource_lease.lease_id == "lease-1"
    injected_tool_message = backend.payloads[1]["messages"][-1]
    assert injected_tool_message["role"] == "tool"
    assert injected_tool_message["tool_call_id"] == "call-echo"
    assert injected_tool_message["name"] == "echo"
    assert json.loads(injected_tool_message["content"]) == {"echo": "hi"}


def test_framework_loop_truncates_tool_observation_for_model_injection_only() -> None:
    large_stdout = "x" * 240
    registry = RuntimeToolRegistry()
    registry.register_local_function(
        ToolSchemaIR(
            name="echo",
            description="Echo text",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            raw_schema={"name": "echo"},
        ),
        lambda arguments, context: {"stdout": large_stdout, "stderr": "", "echo": arguments["text"]},
    )
    backend = _ToolCallingBackend()
    scheduler = FrameworkLoopScheduler(
        backend=backend,
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        max_turns=2,
    )

    result = asyncio.run(
        scheduler.arun(
            session=_session(),
            sample={"messages": [{"role": "user", "content": "hi"}]},
            payload={"max_observation_chars": 80},
            workflow_bundle=_bundle(),
            sandbox_provider=None,
        )
    )

    injected_tool_message = backend.payloads[1]["messages"][-1]
    injected_payload = json.loads(injected_tool_message["content"])
    assert result.agent_output["agent_trace"][0]["output"]["stdout"] == large_stdout
    assert injected_payload["stdout"].startswith("<output truncated:")
    assert "head -n 50" in injected_payload["stdout"]
    assert len(injected_payload["stdout"]) < len(large_stdout)


def test_framework_loop_dynamic_workflow_tools_are_registered_then_runtime_loop_used() -> None:
    registry = RuntimeToolRegistry()
    backend = _FinalBackend()
    scheduler = FrameworkLoopScheduler(
        backend=backend,
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        max_turns=1,
    )

    result = asyncio.run(
        scheduler.arun(
            session=_session(),
            sample={"messages": [{"role": "user", "content": "hi"}]},
            payload={},
            workflow_bundle=_bundle(
                inject_tool_schemas=lambda **_: [
                    {
                        "type": "function",
                        "function": {
                            "name": "legacy",
                            "description": "Legacy tool",
                            "parameters": {"type": "object"},
                        },
                    }
                ]
            ),
            sandbox_provider=None,
        )
    )

    assert result.status == "completed"
    assert backend.payloads[0]["tools"][0]["function"]["name"] == "legacy"
    assert registry.get("legacy") is not None


def test_framework_loop_dynamic_mcp_schema_registers_matching_client_without_overwriting() -> None:
    registry = RuntimeToolRegistry()
    registry.register_local_function(
        ToolSchemaIR(
            name="existing",
            description="Existing tool",
            input_schema={"type": "object"},
            raw_schema={"name": "existing"},
        ),
        lambda arguments, context: {"existing": True},
    )
    mcp_client = _McpClient(tool_name="app__api")
    backend = _FinalBackend()
    scheduler = FrameworkLoopScheduler(
        backend=backend,
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        mcp_clients={"appworld_env": mcp_client},
        max_turns=1,
    )

    asyncio.run(
        scheduler.arun(
            session=_session(benchmark_kit_id="appworld"),
            sample={"messages": [{"role": "user", "content": "hi"}]},
            payload={},
            workflow_bundle=_bundle(
                inject_tool_schemas=lambda **_: [
                    {
                        "name": "app__api",
                        "description": "AppWorld API",
                        "inputSchema": {"type": "object"},
                        "x-gage": {"mcp_client_id": "appworld_env"},
                    },
                    {
                        "name": "existing",
                        "description": "Do not overwrite",
                        "inputSchema": {"type": "object"},
                        "x-gage": {"mcp_client_id": "appworld_env"},
                    },
                ]
            ),
            sandbox_provider=None,
        )
    )

    app_entry = registry.get("app__api")
    existing_entry = registry.get("existing")
    assert app_entry is not None
    assert app_entry.provider_kind == "mcp"
    assert app_entry.executor is mcp_client
    assert existing_entry is not None
    assert existing_entry.provider_kind == "local_function"


def test_framework_loop_runtime_tooling_writes_tool_trace_events(tmp_path: Path) -> None:
    registry = RuntimeToolRegistry()
    registry.register_local_function(
        ToolSchemaIR(
            name="echo",
            description="Echo text",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            raw_schema={"name": "echo"},
        ),
        lambda arguments, context: {"echo": arguments["text"]},
    )
    backend = _ToolCallingBackend()
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    scheduler = FrameworkLoopScheduler(
        backend=backend,
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        max_turns=2,
    )

    asyncio.run(
        scheduler.arun(
            session=_session(),
            sample={"messages": [{"role": "user", "content": "hi"}]},
            payload={"artifact_sink": sink},
            workflow_bundle=_bundle(),
            sandbox_provider=None,
        )
    )

    trace_path = tmp_path / "run-1" / "artifacts/task-1/sample-1/trials/trial_0001/infra/trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["event_type"] for event in events]

    assert event_types == [
        "model.request",
        "model.response",
        "tool.call.raw",
        "tool.call.normalized",
        "tool.result",
        "tool.result.injected",
        "model.request",
        "model.response",
    ]
    assert events[0]["payload"]["tool_schema_count"] == 1
    assert events[0]["payload"]["raw_request_ref"]["path"].endswith("model_request_turn_1.json")
    assert events[1]["payload"]["tool_call_count"] == 1
    assert events[4]["payload"]["status"] == "success"
    assert events[4]["payload"]["tool_result"]["output_json"] == {"echo": "hi"}
    assert events[5]["payload"]["message"]["tool_call_id"] == "call-echo"


def test_compiled_executor_passes_environment_lease_and_artifact_sink_to_framework_loop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    plan = compile_agent_runtime_plan(agent_runtime_id="swebench_framework_loop")
    plan = replace(
        plan,
        workflow_bundle=_bundle(benchmark_kit_id="swebench"),
        resource_plan=_resource_plan_without_provider_config_resolver(plan),
        verifier_environment_policy="reuse",
        verifier_environment_profile_id=None,
    )
    backend = _RunShellBackend()
    environment_manager = _FakeEnvironmentManager()
    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        static_model_backend=backend,
        environment_manager=environment_manager,
    )
    executor.verifier_runner = _PassingVerifierRunner()

    output = asyncio.run(
        executor.aexecute(
            sample={
                "id": "sample-exec",
                "instruction": "run a shell command",
                "cwd": "/workspace",
                "expected_answer": "done",
            },
            payload={
                "execution_context": {
                    "run_id": "run-exec",
                    "task_id": "task-exec",
                    "sample_id": "sample-exec",
                }
            },
        )
    )

    assert output["answer"] == "done"
    assert ("echo from-tool", 5) in environment_manager.environment.commands
    tool_steps = [step for step in output["agent_trace"] if step["trace_role"] == "tool"]
    assert tool_steps[0]["status"] == "success"
    assert tool_steps[0]["output"]["stdout"] == "from env\n"

    trace_path = tmp_path / "run-exec/artifacts/task-exec/sample-exec/trials/trial_0001/infra/trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["event_type"] for event in events]
    assert event_types == [
        "trial.start",
        "environment.acquire",
        "client.environment_handle.projected",
        "model.request",
        "model.response",
        "tool.call.raw",
        "tool.call.normalized",
        "tool.result",
        "tool.result.injected",
        "model.request",
        "model.response",
        "verifier.result",
        "trial.end",
    ]
    by_type = {event["event_type"]: event for event in events}
    assert by_type["trial.start"]["payload"]["scheduler_type"] == "framework_loop"
    assert by_type["environment.acquire"]["payload"]["role"] == "scheduler"
    assert "environment_handle" in by_type["client.environment_handle.projected"]["payload"]
    assert by_type["tool.result"]["payload"]["status"] == "success"
    assert by_type["tool.result"]["payload"]["tool_result"]["output_json"]["stdout"] == "from env\n"
    assert by_type["verifier.result"]["actor"] == "verifier"
    assert by_type["verifier.result"]["payload"]["metric"]["score"] == 1.0
    assert by_type["verifier.result"]["artifact_refs"]
    assert by_type["trial.end"]["payload"]["status"] == "completed"
    assert by_type["trial.end"]["payload"]["failure"] is None


def test_compiled_executor_resets_and_binds_mcp_process_for_trial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    plan = compile_agent_runtime_plan(agent_runtime_id="swebench_framework_loop")
    plan = replace(
        plan,
        workflow_bundle=_bundle(benchmark_kit_id="swebench"),
        resource_plan=_resource_plan_without_provider_config_resolver(plan),
        agent_config={"tooling": {"mcp_servers": ["local_mcp"]}},
        judge_binding=replace(plan.judge_binding, judge_mode="disabled"),
    )
    backend = _McpToolBackend()
    environment_manager = _FakeEnvironmentManager()
    mcp_client = _McpClient(tool_name="mcp_lookup")
    mcp_process = McpServerProcess(server_id="local_mcp", client=mcp_client)
    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        static_model_backend=backend,
        mcp_clients={"local_mcp": mcp_process},
        environment_manager=environment_manager,
    )

    output = asyncio.run(
        executor.aexecute(
            sample={"id": "sample-mcp", "instruction": "use mcp", "cwd": "/workspace"},
            payload={
                "execution_context": {
                    "run_id": "run-mcp",
                    "task_id": "task-mcp",
                    "sample_id": "sample-mcp",
                }
            },
        )
    )

    assert output["answer"] == "done"
    assert mcp_process.owner_trial_id == "trial_0001"
    assert mcp_process.reset_count == 1
    assert mcp_client.reset_count == 1
    assert mcp_client.calls == [("mcp_lookup", {"q": "refund"})]
    registry_entry = executor.compiled_plan.tool_registry.get("mcp_lookup")
    assert registry_entry.executor is mcp_client


def test_framework_loop_tau2_text_dialect_reaches_router_with_schema_coercion() -> None:
    observed_arguments: list[dict[str, Any]] = []
    registry = RuntimeToolRegistry()
    registry.register_local_function(
        ToolSchemaIR(
            name="respond",
            description="Respond to the user",
            input_schema={
                "type": "object",
                "required": ["message", "urgent", "count"],
                "properties": {
                    "message": {"type": "string"},
                    "urgent": {"type": "boolean"},
                    "count": {"type": "integer"},
                },
            },
            raw_schema={"name": "respond"},
        ),
        lambda arguments, context: _record_tau2_arguments(arguments, observed_arguments),
    )
    backend = _Tau2TextBackend()
    scheduler = FrameworkLoopScheduler(
        backend=backend,
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        max_turns=2,
    )

    result = asyncio.run(
        scheduler.arun(
            session=_session(benchmark_kit_id="tau2"),
            sample={"messages": [{"role": "user", "content": "hi"}]},
            payload={"tool_choice": "none"},
            workflow_bundle=_bundle(benchmark_kit_id="tau2", bundle_id="tau2.framework_loop"),
            sandbox_provider=None,
        )
    )

    assert result.status == "completed"
    assert observed_arguments == [{"message": "hello", "urgent": True, "count": 12}]
    injected_tool_message = backend.payloads[1]["messages"][-1]
    assert injected_tool_message["role"] == "tool"
    assert injected_tool_message["name"] == "respond"


def test_framework_loop_tau2_required_tool_retry_budget_exhausts_on_empty_response() -> None:
    backend = _Tau2EmptyTextBackend()
    registry = build_tau2_tool_registry()
    scheduler = FrameworkLoopScheduler(
        backend=backend,
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        max_turns=8,
    )

    result = asyncio.run(
        scheduler.arun(
            session=_session(benchmark_kit_id="tau2"),
            sample={"messages": [{"role": "user", "content": "hi"}]},
            payload={"tool_call_retry_budget": 3},
            workflow_bundle=_bundle(benchmark_kit_id="tau2", bundle_id="tau2.framework_loop"),
            sandbox_provider=None,
        )
    )

    assert len(backend.payloads) == 3
    assert result.status == "failed"
    assert result.agent_output["loop_exit_reason"] == "tool_call_retry_budget"
    assert result.agent_output["failure_code"] == "client_execution.tool_retry_budget_exhausted"


def test_framework_loop_required_tool_review_required_does_not_complete_submission() -> None:
    registry = RuntimeToolRegistry()
    registry.register_local_function(
        ToolSchemaIR(
            name="submit_patch_tool",
            description="Submit patch",
            input_schema={"type": "object"},
            raw_schema={"name": "submit_patch_tool"},
        ),
        lambda arguments, context: {
            "status": "review_required",
            "final_answer": "this review checkpoint must not terminate the run",
        },
    )
    backend = _SubmitReviewThenProseBackend()
    scheduler = FrameworkLoopScheduler(
        backend=backend,
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        max_turns=3,
    )

    result = asyncio.run(
        scheduler.arun(
            session=_session(benchmark_kit_id="swebench"),
            sample={"messages": [{"role": "user", "content": "fix it"}]},
            payload={"required_tool": "submit_patch_tool", "tool_call_retry_budget": 1},
            workflow_bundle=_bundle(benchmark_kit_id="swebench", bundle_id="swebench.framework_loop"),
            sandbox_provider=None,
        )
    )

    assert len(backend.payloads) == 2
    assert result.status == "failed"
    assert result.agent_output["answer"] == ""
    assert result.agent_output["loop_exit_reason"] == "tool_call_retry_budget"
    assert result.agent_output["failure_reason"] == "required_tool_missing"


def test_framework_loop_tau2_plain_text_response_advances_user_simulator(tmp_path: Path) -> None:
    lease = _Tau2ScriptedRespondLease()
    backend = _Tau2ConversationBackend()
    registry = build_tau2_tool_registry()
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    scheduler = FrameworkLoopScheduler(
        backend=backend,
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        max_turns=3,
    )

    result = asyncio.run(
        scheduler.arun(
            session=_session(benchmark_kit_id="tau2"),
            sample={"messages": [{"role": "user", "content": "hi"}]},
            payload={"environment_lease": lease, "artifact_sink": sink},
            workflow_bundle=_bundle(benchmark_kit_id="tau2", bundle_id="tau2.framework_loop"),
            sandbox_provider=None,
        )
    )

    assert result.status == "completed"
    assert result.agent_output["answer"] == "###STOP###"
    assert lease.calls == [
        ("respond", {"message": "Could you tell me which line?"}),
        ("respond", {"message": "###STOP###"}),
    ]
    assert backend.payloads[1]["messages"][-2:] == [
        {"role": "assistant", "content": "Could you tell me which line?"},
        {"role": "user", "content": "The line ending in 1234."},
    ]
    trace_path = tmp_path / "run-1/artifacts/task-1/sample-1/trials/trial_0001/infra/trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert "user.message.injected" in [event["event_type"] for event in events]


def test_framework_loop_tau2_respond_tool_uses_environment_lease() -> None:
    lease = _Tau2EnvironmentLease()
    backend = _Tau2TextBackend()
    registry = build_tau2_tool_registry()
    scheduler = FrameworkLoopScheduler(
        backend=backend,
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        max_turns=2,
    )

    result = asyncio.run(
        scheduler.arun(
            session=_session(benchmark_kit_id="tau2"),
            sample={"messages": [{"role": "user", "content": "hi"}]},
            payload={"environment_lease": lease},
            workflow_bundle=_bundle(benchmark_kit_id="tau2", bundle_id="tau2.framework_loop"),
            sandbox_provider=None,
        )
    )

    assert result.status == "completed"
    assert result.agent_output["answer"] == "hello"
    assert lease.calls
    assert lease.calls[0][0] == "respond"
    assert lease.calls[0][1]["message"] == "hello"


def test_framework_loop_tau2_repeated_nonterminal_tools_exhaust_max_turns() -> None:
    lease = _Tau2NonTerminalEnvironmentLease()
    backend = _Tau2TextBackend()
    registry = build_tau2_tool_registry()
    scheduler = FrameworkLoopScheduler(
        backend=backend,
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        max_turns=1,
    )

    result = asyncio.run(
        scheduler.arun(
            session=_session(benchmark_kit_id="tau2"),
            sample={"messages": [{"role": "user", "content": "hi"}]},
            payload={"environment_lease": lease},
            workflow_bundle=_bundle(benchmark_kit_id="tau2", bundle_id="tau2.framework_loop"),
            sandbox_provider=None,
        )
    )

    assert result.status == "failed"
    assert result.agent_output["loop_exit_reason"] == "max_turns"
    assert result.agent_output["failure_code"] == "client_execution.tool_retry_budget_exhausted"
    assert lease.calls == [("respond", {"message": "hello", "urgent": "true", "count": "12"})]


def test_framework_loop_tooling_parse_error_preserves_failure_code() -> None:
    scheduler = FrameworkLoopScheduler(
        backend=_InvalidJsonBackend(),
        tool_router=ToolRouter(RuntimeToolRegistry()),
        tool_registry=RuntimeToolRegistry(),
        max_turns=1,
    )

    try:
        asyncio.run(
            scheduler.arun(
                session=_session(),
                sample={"messages": [{"role": "user", "content": "hi"}]},
                payload={},
                workflow_bundle=_bundle(),
                sandbox_provider=None,
            )
        )
    except FailureEnvelopeError as exc:
        failure = exc.failure
    else:  # pragma: no cover - assertion path
        raise AssertionError("framework_loop should surface provider parse failure")

    assert failure.failure_code == "client_execution.tool_protocol_parse_error"
    assert failure.details["tooling_code"] == "client_execution.tool_protocol_parse_error"


def test_framework_loop_model_request_timeout_uses_specific_failure_code() -> None:
    scheduler = FrameworkLoopScheduler(
        backend=_TimeoutBackend(),
        tool_router=ToolRouter(RuntimeToolRegistry()),
        tool_registry=RuntimeToolRegistry(),
        max_turns=1,
    )

    try:
        asyncio.run(
            scheduler.arun(
                session=_session(),
                sample={"messages": [{"role": "user", "content": "hi"}]},
                payload={},
                workflow_bundle=_bundle(),
                sandbox_provider=None,
            )
        )
    except FailureEnvelopeError as exc:
        failure = exc.failure
    else:  # pragma: no cover - assertion path
        raise AssertionError("framework_loop should surface model request timeout")

    assert failure.failure_code == "client_execution.model_request_timeout"


def test_framework_loop_model_provider_timeout_class_uses_specific_failure_code() -> None:
    scheduler = FrameworkLoopScheduler(
        backend=_LitellmTimeoutBackend(),
        tool_router=ToolRouter(RuntimeToolRegistry()),
        tool_registry=RuntimeToolRegistry(),
        max_turns=1,
    )

    try:
        asyncio.run(
            scheduler.arun(
                session=_session(),
                sample={"messages": [{"role": "user", "content": "hi"}]},
                payload={},
                workflow_bundle=_bundle(),
                sandbox_provider=None,
            )
        )
    except FailureEnvelopeError as exc:
        failure = exc.failure
    else:  # pragma: no cover - assertion path
        raise AssertionError("framework_loop should surface provider timeout classes")

    assert failure.failure_code == "client_execution.model_request_timeout"
    assert failure.details["exception_type"] == "_LitellmTimeout"


def test_framework_loop_cost_limit_stops_expensive_run() -> None:
    scheduler = FrameworkLoopScheduler(
        backend=_CostBackend(),
        tool_router=ToolRouter(RuntimeToolRegistry()),
        tool_registry=RuntimeToolRegistry(),
        max_turns=1,
    )

    result = asyncio.run(
        scheduler.arun(
            session=_session(),
            sample={"messages": [{"role": "user", "content": "hi"}]},
            payload={"cost_limit_usd": 5.0},
            workflow_bundle=_bundle(),
            sandbox_provider=None,
        )
    )

    assert result.status == "failed"
    assert result.agent_output["failure_code"] == "client_execution.cost_limit_exceeded"
    assert result.agent_output["usage"]["cost_usd"] == 6.0


def test_framework_loop_default_turn_limit_is_swebench_friendly() -> None:
    scheduler = FrameworkLoopScheduler(
        backend=_FinalBackend(),
        tool_router=ToolRouter(RuntimeToolRegistry()),
        tool_registry=RuntimeToolRegistry(),
    )

    assert scheduler._max_turns == 150


def test_framework_loop_tool_result_injection_error_preserves_failure_code() -> None:
    registry = RuntimeToolRegistry()
    registry.register_local_function(
        ToolSchemaIR(
            name="echo",
            description="Echo text",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            raw_schema={"name": "echo"},
        ),
        lambda arguments, context: object(),
    )
    scheduler = FrameworkLoopScheduler(
        backend=_ToolCallingBackend(),
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        max_turns=2,
    )

    try:
        asyncio.run(
            scheduler.arun(
                session=_session(),
                sample={"messages": [{"role": "user", "content": "hi"}]},
                payload={},
                workflow_bundle=_bundle(),
                sandbox_provider=None,
            )
        )
    except FailureEnvelopeError as exc:
        failure = exc.failure
    else:  # pragma: no cover - assertion path
        raise AssertionError("framework_loop should surface tool result injection failure")

    assert failure.failure_code == "client_execution.tool_result_injection_failed"
    assert failure.details["tooling_code"] == "client_execution.tool_result_injection_failed"


def test_framework_loop_rejects_direct_mcp_client_skill_manifest_or_provider_raw_schema() -> None:
    for leaked in (_McpClient(), {"skill": "manifest", "tools": []}, _RawProviderSchema()):
        scheduler = FrameworkLoopScheduler(
            backend=_FinalBackend(),
            tool_router=ToolRouter(RuntimeToolRegistry()),
            tool_registry=leaked,
            max_turns=1,
        )

        result = asyncio.run(
            scheduler.arun(
                session=_session(),
                sample={"messages": [{"role": "user", "content": "hi"}]},
                payload={},
                workflow_bundle=_bundle(),
                sandbox_provider=None,
            )
        )

        assert result.status == "failed"
        assert result.failure is not None
        assert result.failure.failure_code == "client_execution.tool_schema_invalid"


def test_framework_loop_source_does_not_import_or_construct_legacy_agent_loop() -> None:
    import gage_eval.agent_runtime.schedulers.framework_loop as module

    source = inspect.getsource(module)

    assert "role.agent.loop" not in source
    assert "AgentLoop(" not in source


def test_compiled_framework_loop_plans_have_kit_tool_registry_entries() -> None:
    expected = {
        "swebench_framework_loop": {"run_shell", "submit_patch_tool"},
        "tau2_framework_loop": {"respond"},
    }

    for runtime_id, expected_names in expected.items():
        plan = compile_agent_runtime_plan(agent_runtime_id=runtime_id)
        assert isinstance(plan.tool_registry, RuntimeToolRegistry)
        assert expected_names.issubset(set(plan.tool_registry.entries()))
        assert plan.tool_registry.project_tool_schemas()


def test_resolver_builds_runtime_framework_loop_with_new_tool_router() -> None:
    registry = RuntimeToolRegistry()
    plan = replace(compile_agent_runtime_plan(agent_runtime_id="swebench_framework_loop"), tool_registry=registry)

    executor = build_compiled_runtime_executor(compiled_plan=plan, agent_backend=_FinalBackend())

    scheduler = executor.compiled_plan.scheduler_handle
    assert isinstance(scheduler, FrameworkLoopScheduler)
    assert isinstance(scheduler._tool_router, ToolRouter)
    assert scheduler._tool_registry is registry


def _record_dispatch(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
    dispatches: list[tuple[dict[str, Any], ToolExecutionContext]],
) -> dict[str, Any]:
    dispatches.append((dict(arguments), context))
    return {"echo": arguments["text"]}


def _resource_plan_without_provider_config_resolver(plan) -> dict[str, Any]:
    resource_plan = dict(plan.resource_plan or {})
    resource_plan.pop("provider_config_resolver", None)
    return resource_plan


def _record_tau2_arguments(arguments: dict[str, Any], observed_arguments: list[dict[str, Any]]) -> dict[str, Any]:
    observed_arguments.append(dict(arguments))
    return {"ok": True}


def _bundle(**overrides: Any) -> SchedulerWorkflowBundle:
    values = {
        "bundle_id": "demo.framework_loop",
        "benchmark_kit_id": "demo",
        "scheduler_type": "framework_loop",
    }
    values.update(overrides)
    return SchedulerWorkflowBundle(**values)


def _session(*, benchmark_kit_id: str = "demo") -> AgentRuntimeSession:
    return AgentRuntimeSession(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id=benchmark_kit_id,
        scheduler_type="framework_loop",
        resource_lease=ResourceLease(
            lease_id="lease-1",
            resource_kind="local_process",
            profile_id="demo",
            lifecycle="per_sample",
        ),
    )
