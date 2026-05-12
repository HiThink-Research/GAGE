from __future__ import annotations

import inspect
import json
import re
import shlex
import time
from typing import Any

from gage_eval.environment.contracts import ExecResult
from gage_eval.agent_runtime.tooling.contracts import ToolCallIR, ToolExecutionContext, ToolResultIR, ToolingError
from gage_eval.agent_runtime.tooling.registry import RuntimeToolEntry, RuntimeToolRegistry


class ToolRouter:
    """Dispatch ToolCallIR through registered runtime tool providers."""

    def __init__(self, registry: RuntimeToolRegistry) -> None:
        self._registry = registry

    async def dispatch(self, call: ToolCallIR, context: ToolExecutionContext) -> ToolResultIR:
        start = time.perf_counter()
        entry = self._registry.get(call.name)
        if entry is None:
            return _failure_result(
                call,
                code="client_execution.tool_router.not_found",
                message=f"tool not found: {call.name}",
                latency_ms=_latency_ms(start),
            )

        try:
            arguments = call.arguments()
            arguments = _validate_and_coerce_arguments(entry.schema.input_schema, arguments)
            output = await self._dispatch_entry(entry, call=call, arguments=arguments, context=context)
            return ToolResultIR(
                call_id=call.call_id,
                name=call.name,
                provider=call.provider,
                status="success",
                output_text=_output_text(output),
                output_json=output,
                raw_output=output,
                metadata={"latency_ms": _latency_ms(start)},
            )
        except ToolingError as exc:
            return _failure_result(call, code=exc.code, message=str(exc), details=exc.details, latency_ms=_latency_ms(start))
        except Exception as exc:
            return _failure_result(
                call,
                code="client_execution.tool_argument_invalid",
                message=str(exc),
                details={"error_type": exc.__class__.__name__},
                latency_ms=_latency_ms(start),
            )

    @staticmethod
    def raise_retry_budget_exhausted(*, required_tool: str) -> None:
        raise ToolingError(
            "client_execution.tool_retry_budget_exhausted",
            f"required tool retry budget exhausted: {required_tool}",
            details={"required_tool": required_tool},
        )

    async def _dispatch_entry(
        self,
        entry: RuntimeToolEntry,
        *,
        call: ToolCallIR,
        arguments: Any,
        context: ToolExecutionContext,
    ) -> Any:
        if entry.provider_kind == "environment":
            return await _dispatch_environment(call.name, arguments, context)
        if entry.provider_kind == "human":
            return await _dispatch_human(entry.executor, call.name, arguments, context)
        if entry.provider_kind == "mcp":
            return await _dispatch_mcp(entry.executor, call.name, arguments)
        if not callable(entry.executor):
            raise ToolingError("client_execution.tool_router.not_found", f"tool executor missing: {call.name}")
        return await _call_local_function(entry.executor, arguments, context)


async def _dispatch_environment(name: str, arguments: Any, context: ToolExecutionContext) -> Any:
    lease = context.environment_lease
    if lease is None:
        raise ToolingError(
            "client_execution.tool_router.environment_unavailable",
            "environment tool requires ToolExecutionContext.environment_lease",
            details={"tool": name},
        )
    if not isinstance(arguments, dict):
        raise ToolingError("client_execution.tool_argument_invalid", "environment tool arguments must be a dict")
    if name == "run_shell":
        kwargs = _filter_kwargs(arguments, {"env", "cwd", "timeout_s", "user", "shell"})
        if "timeout_s" in kwargs and kwargs["timeout_s"] is not None:
            kwargs["timeout_s"] = int(kwargs["timeout_s"])
        return _exec_result_payload(
            await _await_call(
                lease.exec(
                    str(arguments.get("command") or arguments.get("cmd") or ""),
                    **kwargs,
                )
            )
        )
    if name == "read_file":
        path = _require_absolute_path(str(arguments.get("path") or ""), tool=name)
        content = await _await_call(
            lease.read_file(
                path,
                **_filter_kwargs(arguments, {"max_bytes"}),
            )
        )
        return {"content": content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)}
    if name == "write_file":
        path = _require_absolute_path(str(arguments.get("path") or ""), tool=name)
        await _await_call(lease.write_file(path, arguments.get("content") or ""))
        return {"status": "ok"}
    if name == "replace_in_file":
        path = _require_absolute_path(str(arguments.get("path") or ""), tool=name)
        pattern = str(arguments.get("pattern") or "")
        replacement = str(arguments.get("replacement") or "")
        content = await _await_call(lease.read_file(path))
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
        count = arguments.get("count")
        updated = re.sub(pattern, replacement, text, count=0 if count is None else int(count))
        await _await_call(lease.write_file(path, updated))
        return {"status": "ok"}
    if name == "str_replace_editor":
        return await _str_replace_editor(arguments, lease, context)
    if name == "view_file_window":
        return await _view_file_window(arguments, lease)
    if name == "find_in_repo":
        path = str(arguments.get("path") or ".")
        pattern = str(arguments.get("pattern") or "")
        max_results = max(1, min(1000, int(arguments.get("max_results") or 200)))
        command = (
            "grep -RIn --exclude-dir=.git -- "
            f"{shlex.quote(pattern)} {shlex.quote(path)} | head -n {max_results}"
        )
        return _exec_result_payload(await _await_call(lease.exec(command, timeout_s=arguments.get("timeout_s"))))
    if name == "find_file":
        path = str(arguments.get("path") or ".")
        pattern = str(arguments.get("pattern") or arguments.get("name") or "")
        max_results = max(1, min(1000, int(arguments.get("max_results") or 200)))
        command = f"find {shlex.quote(path)} -type f -name {shlex.quote(pattern)} | head -n {max_results}"
        return _exec_result_payload(await _await_call(lease.exec(command, timeout_s=arguments.get("timeout_s"))))
    if name == "submit_patch_tool":
        if not arguments.get("force") and not context.metadata.get("submit_patch_acknowledged"):
            context.metadata["submit_patch_acknowledged"] = True
            return {
                "status": "review_required",
                "checklist": [
                    "Did you revert any modifications to test files?",
                    "Did you remove temporary repro scripts you do not want in the patch?",
                    "Are all changes minimal and necessary?",
                ],
                "note": "Call submit_patch_tool again to confirm submission.",
            }
        command = "git add -A && git diff --cached --binary --no-color"
        payload = _exec_result_payload(await _await_call(lease.exec(command, timeout_s=arguments.get("timeout_s"))))
        stdout = str(payload.get("stdout") or "")
        payload["patch_content"] = stdout
        payload["final_answer"] = stdout
        return payload
    for method_name in ("upload_file", "download_file", "upload_dir", "download_dir"):
        if name == method_name and callable(getattr(lease, method_name, None)):
            method = getattr(lease, method_name)
            await _await_call(method(**arguments))
            return {"status": "ok"}
    for method_name in ("call_tool", "exec_tool"):
        method = getattr(lease, method_name, None)
        if callable(method):
            return await _await_call(method(name, arguments))
    raise ToolingError(
        "client_execution.tool_router.not_found",
        f"environment tool is not mapped: {name}",
        details={"tool": name},
    )


async def _dispatch_human(gateway: Any, name: str, arguments: Any, context: ToolExecutionContext) -> Any:
    request = getattr(gateway, "request", None)
    if not callable(request):
        raise ToolingError(
            "client_execution.tool_router.human_gateway_unavailable",
            "human tool requires a HumanGateway executor",
            details={"tool": name},
        )
    if isinstance(arguments, dict):
        question = str(
            arguments.get("question")
            or arguments.get("prompt")
            or arguments.get("input")
            or arguments.get("message")
            or ""
        )
    else:
        question = str(arguments)
    response = await _await_call(
        request(
            question,
            metadata={"tool": name, "run_id": context.run_id, "trial_id": context.trial_id},
        )
    )
    return {"response": response}


async def _dispatch_mcp(client: Any, name: str, arguments: Any) -> Any:
    call_tool = getattr(client, "call_tool", None)
    if not callable(call_tool):
        raise ToolingError("client_execution.tool_router.not_found", "MCP client does not expose call_tool")
    return await _await_call(call_tool(name, arguments))


async def _call_local_function(function: Any, arguments: Any, context: ToolExecutionContext) -> Any:
    signature = inspect.signature(function)
    if len(signature.parameters) >= 2:
        return await _await_call(function(arguments, context))
    return await _await_call(function(arguments))


def _failure_result(
    call: ToolCallIR,
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    latency_ms: float,
) -> ToolResultIR:
    return ToolResultIR(
        call_id=call.call_id,
        name=call.name,
        provider=call.provider,
        status="error",
        output_text=message,
        output_json={
            "failure_code": code,
            "error": message,
            "details": details or {},
        },
        raw_output={"error": message, "failure_code": code},
        metadata={"latency_ms": latency_ms},
    )


def _latency_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=True, separators=(",", ":"))
    except TypeError:
        return str(output)


async def _await_call(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _exec_result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, ExecResult):
        return result.model_dump(mode="python")
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="python")
    if isinstance(result, dict):
        return dict(result)
    return {
        "command": getattr(result, "command", ""),
        "exit_code": getattr(result, "exit_code", None),
        "stdout": getattr(result, "stdout", ""),
        "stderr": getattr(result, "stderr", ""),
    }


async def _str_replace_editor(arguments: dict[str, Any], lease: Any, context: ToolExecutionContext) -> dict[str, Any]:
    command = str(arguments.get("command") or "view")
    path = _require_absolute_path(str(arguments.get("path") or ""), tool="str_replace_editor")
    if command == "view":
        content = await _read_text(lease, path)
        start_line, end_line = _view_range(arguments.get("view_range"), line_count=len(content.splitlines()))
        return {
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "content": _format_line_window(content, start_line=start_line, end_line=end_line),
        }
    if command == "create":
        file_text = str(arguments.get("file_text") if arguments.get("file_text") is not None else "")
        await _await_call(lease.write_file(path, file_text))
        return {"status": "ok", "path": path}
    if command == "str_replace":
        old_str = str(arguments.get("old_str") if arguments.get("old_str") is not None else "")
        new_str = str(arguments.get("new_str") if arguments.get("new_str") is not None else "")
        if not old_str:
            raise ToolingError("client_execution.tool_argument_invalid", "str_replace_editor requires old_str")
        content = await _read_text(lease, path)
        replacement_count = content.count(old_str)
        if replacement_count == 0:
            raise ToolingError(
                "client_execution.tool_argument_invalid",
                "old_str was not found in file",
                details={"path": path},
            )
        if replacement_count > 1:
            raise ToolingError(
                "client_execution.tool_argument_invalid",
                f"old_str matched {replacement_count} times; provide a more specific snippet",
                details={"path": path, "match_count": replacement_count},
            )
        _push_editor_snapshot(context, path, content)
        await _await_call(lease.write_file(path, content.replace(old_str, new_str, 1)))
        return {"status": "ok", "path": path, "replacement_count": 1}
    if command == "insert":
        insert_line = int(arguments.get("insert_line") or 0)
        new_str = str(arguments.get("new_str") if arguments.get("new_str") is not None else "")
        content = await _read_text(lease, path)
        lines = content.splitlines(keepends=True)
        index = max(0, min(len(lines), insert_line))
        if new_str and not new_str.endswith("\n"):
            new_str += "\n"
        lines.insert(index, new_str)
        _push_editor_snapshot(context, path, content)
        await _await_call(lease.write_file(path, "".join(lines)))
        return {"status": "ok", "path": path, "insert_line": insert_line}
    if command == "undo_edit":
        previous = _pop_editor_snapshot(context, path)
        if previous is None:
            raise ToolingError(
                "client_execution.tool_argument_invalid",
                "no previous edit is available to undo for this file",
                details={"path": path},
            )
        await _await_call(lease.write_file(path, previous))
        return {"status": "ok", "path": path}
    raise ToolingError(
        "client_execution.tool_argument_invalid",
        f"unsupported str_replace_editor command: {command}",
        details={"command": command},
    )


async def _view_file_window(arguments: dict[str, Any], lease: Any) -> dict[str, Any]:
    path = _require_absolute_path(str(arguments.get("path") or ""), tool="view_file_window")
    start_line = max(1, int(arguments.get("start_line") or 1))
    line_count = max(1, min(1000, int(arguments.get("line_count") or 100)))
    content = await _read_text(lease, path)
    end_line = min(len(content.splitlines()), start_line + line_count - 1)
    return {
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "content": _format_line_window(content, start_line=start_line, end_line=end_line),
    }


async def _read_text(lease: Any, path: str) -> str:
    content = await _await_call(lease.read_file(path))
    return content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)


def _require_absolute_path(path: str, *, tool: str) -> str:
    if not path:
        raise ToolingError("client_execution.tool_argument_invalid", f"{tool} requires path")
    if not path.startswith("/"):
        raise ToolingError(
            "client_execution.tool_argument_invalid",
            f"path must be absolute (e.g. /app/src/foo.py); got {path!r}. "
            "Use the repository root path shown in the system prompt.",
            details={"path": path},
        )
    return path


def _push_editor_snapshot(context: ToolExecutionContext, path: str, content: str) -> None:
    history = context.metadata.setdefault("str_replace_editor_history", {})
    if not isinstance(history, dict):
        history = {}
        context.metadata["str_replace_editor_history"] = history
    stack = history.setdefault(path, [])
    if not isinstance(stack, list):
        stack = []
        history[path] = stack
    stack.append(content)


def _pop_editor_snapshot(context: ToolExecutionContext, path: str) -> str | None:
    history = context.metadata.get("str_replace_editor_history")
    if not isinstance(history, dict):
        return None
    stack = history.get(path)
    if not isinstance(stack, list) or not stack:
        return None
    previous = stack.pop()
    return previous if isinstance(previous, str) else None


def _view_range(raw_range: Any, *, line_count: int) -> tuple[int, int]:
    if isinstance(raw_range, list) and len(raw_range) >= 2:
        start_line = max(1, int(raw_range[0]))
        end_line = max(start_line, int(raw_range[1]))
        return start_line, min(max(line_count, 1), end_line)
    end_line = min(max(line_count, 1), 100)
    return 1, end_line


def _format_line_window(content: str, *, start_line: int, end_line: int) -> str:
    lines = content.splitlines()
    if not lines or start_line > len(lines):
        return ""
    bounded_end = min(end_line, len(lines))
    return "\n".join(f"{line_no}: {lines[line_no - 1]}" for line_no in range(start_line, bounded_end + 1))


def _filter_kwargs(arguments: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key in allowed and value is not None}


def _validate_and_coerce_arguments(schema: dict[str, Any], arguments: Any) -> dict[str, Any]:
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolingError("client_execution.tool_argument_invalid", "tool arguments must be an object")
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    coerced = dict(arguments)
    for key in required:
        if key not in coerced:
            raise ToolingError(
                "client_execution.tool_argument_invalid",
                f"missing required tool argument: {key}",
                details={"argument": key},
            )
    for key, subschema in properties.items():
        if key not in coerced or not isinstance(subschema, dict):
            continue
        coerced[key] = _coerce_json_schema_value(key, coerced[key], subschema.get("type"))
    return coerced


def _coerce_json_schema_value(key: str, value: Any, expected_type: Any) -> Any:
    if isinstance(expected_type, list):
        expected_type = next((item for item in expected_type if item != "null"), None)
    if expected_type == "string":
        return value if isinstance(value, str) else str(value)
    if expected_type == "integer":
        if isinstance(value, bool):
            raise _argument_type_error(key, "integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
            return int(value)
        raise _argument_type_error(key, "integer")
    if expected_type == "number":
        if isinstance(value, bool):
            raise _argument_type_error(key, "number")
        if isinstance(value, int | float):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise _argument_type_error(key, "number")
    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0", "yes", "no"}:
            return value.strip().lower() in {"true", "1", "yes"}
        raise _argument_type_error(key, "boolean")
    if expected_type == "object" and not isinstance(value, dict):
        raise _argument_type_error(key, "object")
    if expected_type == "array" and not isinstance(value, list):
        raise _argument_type_error(key, "array")
    return value


def _argument_type_error(key: str, expected_type: str) -> ToolingError:
    return ToolingError(
        "client_execution.tool_argument_invalid",
        f"tool argument {key!r} must be {expected_type}",
        details={"argument": key, "expected_type": expected_type},
    )
