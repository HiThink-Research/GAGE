from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.environment.contracts import ExecResult
from gage_eval.agent_runtime.tooling.contracts import (
    TOOLING_FAILURE_CODES,
    ToolCallIR,
    ToolExecutionContext,
    ToolResultIR,
    ToolSchemaIR,
    ToolingError,
)
from gage_eval.agent_runtime.tooling.human_gateway import HumanGateway, HumanRequest
from gage_eval.agent_runtime.tooling.mcp.discovery import discover_mcp_tools
from gage_eval.agent_runtime.tooling.mcp.client import McpServerProcess
from gage_eval.agent_runtime.tooling.registry import RuntimeToolRegistry
from gage_eval.agent_runtime.tooling.router import ToolRouter
from gage_eval.agent_runtime.tooling.skills.policy import SkillPolicy
from gage_eval.agent_runtime.tooling.skills.resolver import SkillManifestResolver


def test_tool_schema_ir_preserves_raw_schema_and_unified_fields() -> None:
    raw_schema = {
        "type": "function",
        "function": {
            "name": "search__openai",
            "description": "Search docs",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        "x-provider": {"source": "openai"},
    }

    schema = ToolSchemaIR.from_provider_schema(raw_schema, provider="openai")

    assert schema.name == "search"
    assert schema.description == "Search docs"
    assert schema.input_schema == raw_schema["function"]["parameters"]
    assert schema.raw_schema == raw_schema
    assert schema.provider_format == "openai"
    assert schema.metadata["provider"] == "openai"
    assert schema.metadata["raw_name"] == "search__openai"


def test_tool_call_normalization_preserves_raw_name_and_arguments_json() -> None:
    raw_call = {
        "id": "provider-call-1",
        "function": {
            "name": "lookup__qwen",
            "arguments": '{"query":"refund status"}',
        },
    }

    call = ToolCallIR.from_provider_call(raw_call, turn_index=3, call_index=2)

    assert call.provider == "openai"
    assert call.call_id == "provider-call-1"
    assert call.name == "lookup"
    assert call.arguments_json == '{"query":"refund status"}'
    assert call.raw_message == raw_call
    assert call.metadata["raw_name"] == "lookup__qwen"


def test_tool_schema_preserves_double_underscore_business_tool_names() -> None:
    schema = ToolSchemaIR.from_provider_schema(
        {
            "name": "api_docs__show_api_doc",
            "inputSchema": {"type": "object"},
        },
        provider="mcp:appworld_env",
    )
    call = ToolCallIR.from_provider_call(
        {
            "id": "call-appworld",
            "function": {
                "name": "api_docs__show_api_doc",
                "arguments": "{}",
            },
        },
        turn_index=1,
        call_index=1,
    )

    assert schema.name == "api_docs__show_api_doc"
    assert schema.metadata == {"provider": "mcp:appworld_env"}
    assert call.name == "api_docs__show_api_doc"
    assert "raw_name" not in call.metadata


def test_tool_call_id_synthesized_when_provider_omits() -> None:
    call = ToolCallIR.from_provider_call(
        {"name": "lookup", "arguments": {"query": "x"}},
        turn_index=4,
        call_index=7,
    )

    assert call.call_id == "call_4_7"
    assert json.loads(call.arguments_json) == {"query": "x"}


def test_tool_result_ir_preserves_status_output_raw_output_and_artifact_refs() -> None:
    result = ToolResultIR(
        call_id="call-1",
        name="lookup",
        provider="openai",
        status="success",
        output_text="42",
        output_json={"answer": "42"},
        raw_output={"provider_payload": {"answer": "42"}},
        artifact_refs=[{"owner": "agent", "name": "stdout.txt"}],
    )

    assert result.status == "success"
    assert result.output_text == "42"
    assert result.output_json == {"answer": "42"}
    assert result.raw_output == {"provider_payload": {"answer": "42"}}
    assert result.artifact_refs == [{"owner": "agent", "name": "stdout.txt"}]


def test_invalid_tool_schema_returns_tool_schema_invalid() -> None:
    registry = RuntimeToolRegistry()

    with pytest.raises(ToolingError) as excinfo:
        registry.register_provider_schema({"type": "function", "function": {"description": "missing name"}})

    assert excinfo.value.code == "client_execution.tool_schema_invalid"


@dataclass
class _EnvironmentLease:
    calls: list[tuple[str, Any]]
    files: dict[str, bytes] = field(default_factory=dict)
    exec_stdout_by_command: dict[str, str] = field(default_factory=dict)

    async def exec(self, command: str, **kwargs: Any) -> ExecResult:
        self.calls.append(("exec", command, kwargs))
        stdout = self.exec_stdout_by_command.get(command, "ok")
        return ExecResult(command=command, exit_code=0, stdout=stdout, stderr="")

    async def read_file(self, path: str, *, max_bytes: int = 16777216) -> bytes:
        self.calls.append(("read_file", path, {"max_bytes": max_bytes}))
        return self.files.get(path, b"file-content")

    async def write_file(self, path: str, content: bytes | str) -> None:
        self.calls.append(("write_file", path, content))
        self.files[path] = content.encode("utf-8") if isinstance(content, str) else bytes(content)


def test_tool_router_environment_tool_requires_lease(tmp_path) -> None:
    registry = RuntimeToolRegistry()
    registry.register_environment_tool(
        ToolSchemaIR(
            name="run_shell",
            description="Run command",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
            raw_schema={"name": "run_shell"},
        )
    )
    router = ToolRouter(registry)
    context = _tool_context(tmp_path, environment_lease=None)

    result = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-1",
                name="run_shell",
                arguments_json='{"command":"pwd"}',
                raw_message={"name": "run_shell"},
            ),
            context,
        ),
    )

    assert result.status == "error"
    assert result.output_json["failure_code"] == "client_execution.tool_router.environment_unavailable"


def test_tool_router_dispatches_environment_tool_through_context_lease(tmp_path) -> None:
    registry = RuntimeToolRegistry()
    registry.register_environment_tool(
        ToolSchemaIR(
            name="run_shell",
            description="Run command",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
            raw_schema={"name": "run_shell"},
        )
    )
    lease = _EnvironmentLease(calls=[])
    router = ToolRouter(registry)

    result = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-1",
                name="run_shell",
                arguments_json='{"command":"pwd","timeout_s":"12"}',
                raw_message={"name": "run_shell"},
            ),
            _tool_context(tmp_path, environment_lease=lease),
        ),
    )

    assert result.status == "success"
    assert result.output_json["stdout"] == "ok"
    assert result.provider == "openai"
    assert lease.calls == [("exec", "pwd", {"timeout_s": 12})]


def test_tool_router_environment_file_tools_use_environment_lease_methods(tmp_path) -> None:
    registry = RuntimeToolRegistry()
    registry.register_environment_tool(
        ToolSchemaIR(
            name="read_file",
            description="Read file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
                "required": ["path"],
            },
            raw_schema={"name": "read_file"},
        )
    )
    registry.register_environment_tool(
        ToolSchemaIR(
            name="write_file",
            description="Write file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            raw_schema={"name": "write_file"},
        )
    )
    lease = _EnvironmentLease(calls=[])
    router = ToolRouter(registry)
    context = _tool_context(tmp_path, environment_lease=lease)

    read_result = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-read",
                name="read_file",
                arguments_json='{"path":"/tmp/a.txt","max_bytes":"8"}',
                raw_message={},
            ),
            context,
        )
    )
    write_result = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-write",
                name="write_file",
                arguments_json='{"path":"/tmp/a.txt","content":"next"}',
                raw_message={},
            ),
            context,
        )
    )

    assert read_result.output_json == {"content": "file-content"}
    assert write_result.output_json == {"status": "ok"}
    assert lease.calls == [
        ("read_file", "/tmp/a.txt", {"max_bytes": 8}),
        ("write_file", "/tmp/a.txt", "next"),
    ]


def test_tool_router_environment_file_tools_reject_relative_paths(tmp_path) -> None:
    registry = RuntimeToolRegistry()
    for name in ("read_file", "write_file", "replace_in_file", "str_replace_editor", "view_file_window"):
        registry.register_environment_tool(
            ToolSchemaIR(
                name=name,
                description=name,
                input_schema={"type": "object", "properties": {}, "required": []},
                raw_schema={"name": name},
            )
        )
    lease = _EnvironmentLease(calls=[], files={"/app/src/app.py": b"one\ntwo\nthree\n"})
    router = ToolRouter(registry)
    context = _tool_context(tmp_path, environment_lease=lease)
    calls = [
        ("read_file", '{"path":"src/app.py"}'),
        ("write_file", '{"path":"src/app.py","content":"print(1)\\n"}'),
        ("replace_in_file", '{"path":"src/app.py","pattern":"one","replacement":"ONE"}'),
        ("str_replace_editor", '{"command":"view","path":"src/app.py","view_range":[2,2]}'),
        ("view_file_window", '{"path":"src/app.py","start_line":3,"line_count":1}'),
    ]

    for tool_name, arguments_json in calls:
        result = _run(
            router.dispatch(
                ToolCallIR(
                    provider="openai",
                    call_id=f"call-{tool_name}",
                    name=tool_name,
                    arguments_json=arguments_json,
                    raw_message={},
                ),
                context,
            )
        )
        assert result.status == "error"
        assert result.output_json["failure_code"] == "client_execution.tool_argument_invalid"
        assert "path must be absolute" in result.output_json["error"]
        assert result.output_json["details"]["path"] == "src/app.py"

    assert lease.calls == []


def test_tool_router_str_replace_editor_supports_windowed_view_create_and_replace(tmp_path) -> None:
    registry = RuntimeToolRegistry()
    registry.register_environment_tool(
        ToolSchemaIR(
            name="str_replace_editor",
            description="Edit files using view/create/str_replace commands",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "path": {"type": "string"},
                    "view_range": {"type": "array"},
                    "file_text": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["command", "path"],
            },
            raw_schema={"name": "str_replace_editor"},
        )
    )
    lease = _EnvironmentLease(calls=[], files={"/workspace/app.py": b"one\ntwo\nthree\n"})
    router = ToolRouter(registry)
    context = _tool_context(tmp_path, environment_lease=lease)

    view = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-view",
                name="str_replace_editor",
                arguments_json='{"command":"view","path":"/workspace/app.py","view_range":[2,3]}',
                raw_message={},
            ),
            context,
        )
    )
    replace = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-replace",
                name="str_replace_editor",
                arguments_json='{"command":"str_replace","path":"/workspace/app.py","old_str":"two","new_str":"TWO"}',
                raw_message={},
            ),
            context,
        )
    )
    create = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-create",
                name="str_replace_editor",
                arguments_json='{"command":"create","path":"/workspace/new.py","file_text":"print(1)\\n"}',
                raw_message={},
            ),
            context,
        )
    )

    assert view.output_json["content"] == "2: two\n3: three"
    assert replace.output_json["replacement_count"] == 1
    assert lease.files["/workspace/app.py"] == b"one\nTWO\nthree\n"
    assert create.output_json == {"status": "ok", "path": "/workspace/new.py"}
    assert lease.files["/workspace/new.py"] == b"print(1)\n"


def test_tool_router_str_replace_editor_supports_undo_edit(tmp_path) -> None:
    registry = RuntimeToolRegistry()
    registry.register_environment_tool(
        ToolSchemaIR(
            name="str_replace_editor",
            description="Edit text files",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "insert_line": {"type": "integer"},
                },
            },
            raw_schema={"name": "str_replace_editor"},
        )
    )
    lease = _EnvironmentLease(calls=[], files={"/workspace/app.py": b"one\ntwo\nthree\n"})
    router = ToolRouter(registry)
    context = _tool_context(tmp_path, environment_lease=lease)

    replace = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-replace",
                name="str_replace_editor",
                arguments_json='{"command":"str_replace","path":"/workspace/app.py","old_str":"two","new_str":"TWO"}',
                raw_message={},
            ),
            context,
        )
    )
    undo = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-undo",
                name="str_replace_editor",
                arguments_json='{"command":"undo_edit","path":"/workspace/app.py"}',
                raw_message={},
            ),
            context,
        )
    )

    assert replace.output_json["status"] == "ok"
    assert undo.output_json == {"status": "ok", "path": "/workspace/app.py"}
    assert lease.files["/workspace/app.py"] == b"one\ntwo\nthree\n"


def test_tool_router_str_replace_editor_undo_requires_previous_edit(tmp_path) -> None:
    registry = RuntimeToolRegistry()
    registry.register_environment_tool(
        ToolSchemaIR(
            name="str_replace_editor",
            description="Edit text files",
            input_schema={"type": "object", "properties": {}},
            raw_schema={"name": "str_replace_editor"},
        )
    )

    result = _run(
        ToolRouter(registry).dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-undo",
                name="str_replace_editor",
                arguments_json='{"command":"undo_edit","path":"/workspace/app.py"}',
                raw_message={},
            ),
            _tool_context(tmp_path, environment_lease=_EnvironmentLease(calls=[])),
        )
    )

    assert result.status == "error"
    assert result.output_json["failure_code"] == "client_execution.tool_argument_invalid"
    assert "no previous edit" in result.output_json["error"]


def test_tool_router_str_replace_editor_rejects_ambiguous_old_str(tmp_path) -> None:
    registry = RuntimeToolRegistry()
    registry.register_environment_tool(
        ToolSchemaIR(
            name="str_replace_editor",
            description="Edit text files",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
            },
            raw_schema={"name": "str_replace_editor"},
        )
    )
    lease = _EnvironmentLease(calls=[], files={"/workspace/app.py": b"two\ntwo\nfoo bar two\n"})
    router = ToolRouter(registry)

    result = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-replace",
                name="str_replace_editor",
                arguments_json='{"command":"str_replace","path":"/workspace/app.py","old_str":"two","new_str":"TWO"}',
                raw_message={},
            ),
            _tool_context(tmp_path, environment_lease=lease),
        )
    )

    assert result.status == "error"
    assert result.output_json["failure_code"] == "client_execution.tool_argument_invalid"
    assert "matched 3 times" in result.output_json["error"]
    assert lease.files["/workspace/app.py"] == b"two\ntwo\nfoo bar two\n"


def test_tool_router_view_file_window_reads_specific_line_range(tmp_path) -> None:
    registry = RuntimeToolRegistry()
    registry.register_environment_tool(
        ToolSchemaIR(
            name="view_file_window",
            description="Read a bounded line window from a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "line_count": {"type": "integer"},
                },
                "required": ["path"],
            },
            raw_schema={"name": "view_file_window"},
        )
    )
    lease = _EnvironmentLease(calls=[], files={"/workspace/app.py": b"one\ntwo\nthree\nfour\n"})

    result = _run(
        ToolRouter(registry).dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-window",
                name="view_file_window",
                arguments_json='{"path":"/workspace/app.py","start_line":"2","line_count":"2"}',
                raw_message={},
            ),
            _tool_context(tmp_path, environment_lease=lease),
        )
    )

    assert result.output_json == {
        "path": "/workspace/app.py",
        "start_line": 2,
        "end_line": 3,
        "content": "2: two\n3: three",
    }


def test_submit_patch_tool_requires_review_then_captures_staged_diff(tmp_path) -> None:
    diff = "diff --git a/foo b/foo\n--- a/foo\n+++ b/foo\n@@ -1 +1 @@\n-old\n+new\n"
    registry = RuntimeToolRegistry()
    registry.register_environment_tool(
        ToolSchemaIR(
            name="submit_patch_tool",
            description="Capture final diff",
            input_schema={
                "type": "object",
                "properties": {"timeout_s": {"type": "integer"}, "force": {"type": "boolean"}},
            },
            raw_schema={"name": "submit_patch_tool"},
        )
    )
    lease = _EnvironmentLease(
        calls=[],
        exec_stdout_by_command={"git add -A && git diff --cached --binary --no-color": diff},
    )
    router = ToolRouter(registry)
    context = _tool_context(tmp_path, environment_lease=lease)

    first = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-submit-1",
                name="submit_patch_tool",
                arguments_json='{"timeout_s":5}',
                raw_message={},
            ),
            context,
        )
    )
    second = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-submit-2",
                name="submit_patch_tool",
                arguments_json='{"timeout_s":5}',
                raw_message={},
            ),
            context,
        )
    )

    assert first.output_json["status"] == "review_required"
    assert "Call submit_patch_tool again" in first.output_json["note"]
    assert lease.calls == [("exec", "git add -A && git diff --cached --binary --no-color", {"timeout_s": 5})]
    assert second.output_json["stdout"] == diff
    assert second.output_json["patch_content"] == diff
    assert second.output_json["final_answer"] == diff


def test_tool_router_unknown_name_returns_not_found(tmp_path) -> None:
    result = _run(
        ToolRouter(RuntimeToolRegistry()).dispatch(
            ToolCallIR(provider="openai", call_id="call-1", name="missing", arguments_json="{}", raw_message={}),
            _tool_context(tmp_path),
        )
    )

    assert result.status == "error"
    assert result.output_json["failure_code"] == "client_execution.tool_router.not_found"


def test_tool_router_dispatches_human_tool_through_runtime_gateway(tmp_path) -> None:
    requests: list[HumanRequest] = []
    gateway = HumanGateway(input_provider=lambda request: requests.append(request) or "approved")
    registry = RuntimeToolRegistry()
    registry.register_human_tool(
        ToolSchemaIR(
            name="ask_human",
            description="Ask a human reviewer",
            input_schema={
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
            raw_schema={"name": "ask_human"},
        ),
        gateway,
    )

    result = _run(
        ToolRouter(registry).dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-human",
                name="ask_human",
                arguments_json='{"question":"Ship this change?"}',
                raw_message={},
            ),
            _tool_context(tmp_path),
        )
    )

    assert result.status == "success"
    assert result.output_json == {"response": "approved"}
    assert requests == [
        HumanRequest(
            question="Ship this change?",
            metadata={"tool": "ask_human", "run_id": "run-1", "trial_id": "trial_0001"},
        )
    ]


def test_tool_argument_invalid_maps_failure_code(tmp_path) -> None:
    registry = RuntimeToolRegistry()
    registry.register_local_function(
        ToolSchemaIR(
            name="echo",
            description="Echo",
            input_schema={"type": "object"},
            raw_schema={"name": "echo"},
        ),
        lambda arguments, context: arguments,
    )

    result = _run(
        ToolRouter(registry).dispatch(
            ToolCallIR(provider="openai", call_id="call-1", name="echo", arguments_json="{not-json", raw_message={}),
            _tool_context(tmp_path),
        )
    )

    assert result.status == "error"
    assert result.output_json["failure_code"] == "client_execution.tool_argument_invalid"


def test_tool_router_validates_required_and_basic_types(tmp_path) -> None:
    registry = RuntimeToolRegistry()
    registry.register_local_function(
        ToolSchemaIR(
            name="typed",
            description="Typed",
            input_schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "dry_run": {"type": "boolean"},
                    "label": {"type": "string"},
                },
                "required": ["count", "dry_run"],
            },
            raw_schema={"name": "typed"},
        ),
        lambda arguments, context: arguments,
    )
    router = ToolRouter(registry)
    context = _tool_context(tmp_path)

    missing = _run(
        router.dispatch(
            ToolCallIR(provider="openai", call_id="call-1", name="typed", arguments_json='{"count":1}', raw_message={}),
            context,
        )
    )
    invalid = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-2",
                name="typed",
                arguments_json='{"count":"abc","dry_run":"true"}',
                raw_message={},
            ),
            context,
        )
    )
    coerced = _run(
        router.dispatch(
            ToolCallIR(
                provider="openai",
                call_id="call-3",
                name="typed",
                arguments_json='{"count":"12","dry_run":"true","label":123}',
                raw_message={},
            ),
            context,
        )
    )

    assert missing.output_json["failure_code"] == "client_execution.tool_argument_invalid"
    assert invalid.output_json["failure_code"] == "client_execution.tool_argument_invalid"
    assert coerced.status == "success"
    assert coerced.output_json == {"count": 12, "dry_run": True, "label": "123"}


def test_tool_result_injection_failure_maps_failure_code() -> None:
    with pytest.raises(ToolingError) as excinfo:
        ToolResultIR.serialize_for_injection(
            ToolResultIR(
                provider="openai",
                call_id="call-1",
                name="echo",
                status="success",
                output_text='{"ok":true}',
                output_json={"ok": True},
            ),
            serializer=lambda _: (_ for _ in ()).throw(RuntimeError("inject failed")),
        )

    assert excinfo.value.code == "client_execution.tool_result_injection_failed"


def test_tool_retry_budget_exhausted_maps_failure_code() -> None:
    with pytest.raises(ToolingError) as excinfo:
        ToolRouter.raise_retry_budget_exhausted(required_tool="submit")

    assert excinfo.value.code == "client_execution.tool_retry_budget_exhausted"


def test_mcp_discovery_failure_maps_failure_code() -> None:
    class FailingClient:
        def list_tools(self) -> list[dict[str, Any]]:
            raise RuntimeError("server gone")

    with pytest.raises(ToolingError) as excinfo:
        discover_mcp_tools(FailingClient(), server_id="demo")

    assert excinfo.value.code == "client_execution.tool_registry.mcp_discovery_failed"


def test_mcp_server_process_tracks_trial_owner_and_reset() -> None:
    class StatefulClient:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    client = StatefulClient()
    process = McpServerProcess(server_id="tools", client=client)

    process.bind_trial("trial_0001")
    process.reset_for_trial("trial_0002")

    assert process.owner_trial_id == "trial_0002"
    assert process.reset_count == 1
    assert client.reset_calls == 1


def test_skill_unavailable_maps_failure_code() -> None:
    resolver = SkillManifestResolver({"available": {"tools": []}})

    with pytest.raises(ToolingError) as excinfo:
        resolver.resolve("missing")

    assert excinfo.value.code == "client_execution.tool_registry.skill_unavailable"


def test_skill_policy_denies_unapproved_skill() -> None:
    resolver = SkillManifestResolver(
        {"available": {"tools": []}},
        policy=SkillPolicy.from_iterable(["different"]),
    )

    with pytest.raises(ToolingError) as excinfo:
        resolver.resolve("available")

    assert excinfo.value.code == "client_execution.tool_registry.skill_policy_denied"


def test_tooling_failure_code_matrix_is_complete() -> None:
    assert TOOLING_FAILURE_CODES == {
        "client_execution.tool_schema_invalid",
        "client_execution.tool_protocol_parse_error",
        "client_execution.tool_protocol_missing_call",
        "client_execution.tool_protocol_missing_call_id",
        "client_execution.tool_router.not_found",
        "client_execution.tool_argument_invalid",
        "client_execution.tool_result_injection_failed",
        "client_execution.tool_retry_budget_exhausted",
        "client_execution.tool_registry.mcp_discovery_failed",
        "client_execution.tool_registry.skill_policy_denied",
        "client_execution.tool_registry.skill_unavailable",
        "client_execution.tool_router.environment_unavailable",
        "client_execution.tool_router.human_gateway_unavailable",
    }


def _tool_context(tmp_path, *, environment_lease: Any | None = None) -> ToolExecutionContext:
    lease = ResourceLease(
        lease_id="lease-1",
        resource_kind="local_process",
        profile_id="tau2_local",
        lifecycle="per_sample",
    )
    return ToolExecutionContext(
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        trial_id="trial_0001",
        resource_lease=lease,
        environment_lease=environment_lease,
        artifact_sink=RuntimeArtifactSink(base_dir=str(tmp_path)),
    )


def _run(awaitable):
    import asyncio

    return asyncio.run(awaitable)
