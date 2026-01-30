"""Lightweight tool router for agent tool calls."""

from __future__ import annotations

import base64
import json
import shlex
import time
from typing import Any, Dict, Optional, Tuple

from gage_eval.role.agent.human_gateway import HumanGateway
from gage_eval.mcp import McpClient
from gage_eval.sandbox.base import BaseSandbox, serialize_exec_result


class ToolRouter:
    """Route tool calls to the sandbox execution entry."""

    def __init__(
        self,
        *,
        default_timeout_s: int = 30,
        mcp_clients: Optional[Dict[str, McpClient]] = None,
        human_gateway: Optional[HumanGateway] = None,
    ) -> None:
        self._default_timeout_s = max(1, int(default_timeout_s))
        self._mcp_clients = mcp_clients or {}
        self._human_gateway = human_gateway

    def execute(
        self,
        tool_call: Dict[str, Any],
        sandbox: Optional[BaseSandbox],
        *,
        tool_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute a tool call via host, MCP, or sandbox.

        Args:
            tool_call: Raw tool call payload from the agent backend.
            sandbox: Sandbox runtime used for default tool execution.
            tool_registry: Optional tool schema registry for routing metadata.

        Returns:
            Tool execution payload with status and timing metadata.
        """

        name, arguments = _normalize_tool_call(tool_call)
        metadata = _resolve_tool_metadata(name, tool_registry, tool_call)
        resolved_tool = _resolve_meta_tool_target(name, arguments, metadata)
        start = time.perf_counter()
        output = {}
        status = "success"
        try:
            output = self._execute_tool(name, arguments, sandbox, tool_registry=tool_registry, tool_call=tool_call)
        except Exception as exc:
            status = "error"
            output = {"error": str(exc)}
        latency_ms = (time.perf_counter() - start) * 1000.0
        result = {
            "name": name,
            "input": arguments,
            "output": _normalize_output(output),
            "status": status,
            "latency_ms": latency_ms,
        }
        final_answer = _resolve_final_answer(result["output"], metadata)
        if status == "success" and final_answer:
            result["final_answer"] = final_answer
        if resolved_tool:
            result["resolved_tool"] = resolved_tool
        return result

    def _execute_tool(
        self,
        name: str,
        arguments: Any,
        sandbox: Optional[BaseSandbox],
        *,
        tool_registry: Optional[Dict[str, Dict[str, Any]]],
        tool_call: Dict[str, Any],
    ) -> Any:
        metadata = _resolve_tool_metadata(name, tool_registry, tool_call)
        if metadata.get("meta_tool"):
            return self._execute_meta_tool(name, arguments, metadata)
        execution = metadata.get("execution")
        if execution == "host":
            if not self._human_gateway:
                raise RuntimeError("human_gateway_unavailable")
            question = _resolve_question(arguments)
            return {"response": self._human_gateway.request(question, metadata={"tool": name})}
        mcp_client_id = metadata.get("mcp_client_id")
        if mcp_client_id:
            client = self._mcp_clients.get(mcp_client_id)
            if client is None:
                raise RuntimeError(f"mcp_client_not_found:{mcp_client_id}")
            return client.call_tool(name, arguments)
        if sandbox is None:
            raise RuntimeError("sandbox_unavailable")
        exec_tool = getattr(sandbox, "exec_tool", None)
        if callable(exec_tool):
            return exec_tool(name, arguments)
        normalized = _normalize_tool_arguments(arguments)
        builtin = _execute_builtin_tool(name, normalized, sandbox, self._default_timeout_s)
        if builtin is not None:
            return builtin
        payload = json.dumps({"tool": name, "arguments": normalized}, ensure_ascii=True)
        return sandbox.exec(payload, timeout=self._default_timeout_s)

    def _execute_meta_tool(
        self,
        name: str,
        arguments: Any,
        metadata: Dict[str, Any],
    ) -> Any:
        app_name = metadata.get("app_name")
        if not app_name and name.startswith("call_"):
            app_name = name.replace("call_", "", 1)
        if not app_name:
            raise RuntimeError("meta_tool_app_missing")
        if not isinstance(arguments, dict):
            raise RuntimeError("meta_tool_payload_invalid")
        endpoint = arguments.get("endpoint") or arguments.get("api_name")
        if not endpoint or not isinstance(endpoint, str):
            raise RuntimeError("meta_tool_endpoint_missing")
        allowed = metadata.get("allowed_endpoints")
        if isinstance(allowed, list) and endpoint not in allowed:
            raise RuntimeError(f"invalid_endpoint:{endpoint}")
        params = _extract_meta_tool_params(arguments)
        tool_name = f"{app_name}__{endpoint}"
        mcp_client_id = metadata.get("mcp_client_id")
        if not mcp_client_id:
            raise RuntimeError("mcp_client_id_missing")
        client = self._mcp_clients.get(mcp_client_id)
        if client is None:
            raise RuntimeError(f"mcp_client_not_found:{mcp_client_id}")
        return client.call_tool(tool_name, dict(params))


def _normalize_tool_call(tool_call: Dict[str, Any]) -> Tuple[str, Any]:
    name = tool_call.get("name")
    arguments: Any = tool_call.get("arguments")
    if "function" in tool_call:
        fn = tool_call.get("function") or {}
        name = fn.get("name") or name
        arguments = fn.get("arguments", arguments)
    if isinstance(arguments, str):
        parsed = _try_parse_json(arguments)
        if parsed is not None:
            arguments = parsed
    return str(name or ""), arguments


def _try_parse_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return None


def _normalize_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict):
        return dict(output)
    return serialize_exec_result(output)


def _normalize_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        parsed = _try_parse_json(arguments)
        if parsed is not None:
            arguments = parsed
        else:
            return {"value": arguments}
    if isinstance(arguments, dict):
        return _unwrap_tool_arguments(arguments)
    return {"value": arguments}


def _resolve_tool_metadata(
    name: str,
    tool_registry: Optional[Dict[str, Dict[str, Any]]],
    tool_call: Dict[str, Any],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if tool_registry and name in tool_registry:
        tool_entry = tool_registry[name]
        if isinstance(tool_entry, dict):
            metadata.update(tool_entry.get("x-gage") or {})
    if isinstance(tool_call, dict):
        metadata.update(tool_call.get("x-gage") or {})
    return metadata


def _resolve_question(arguments: Any) -> str:
    if isinstance(arguments, dict):
        for key in ("question", "prompt", "input", "message"):
            if key in arguments:
                return str(arguments[key])
    return str(arguments)


def _resolve_meta_tool_target(name: str, arguments: Any, metadata: Dict[str, Any]) -> Optional[str]:
    if not metadata.get("meta_tool"):
        return None
    app_name = metadata.get("app_name")
    if not app_name and name.startswith("call_"):
        app_name = name.replace("call_", "", 1)
    if not app_name:
        return None
    if not isinstance(arguments, dict):
        return None
    endpoint = arguments.get("endpoint") or arguments.get("api_name")
    if not endpoint or not isinstance(endpoint, str):
        return None
    allowed = metadata.get("allowed_endpoints")
    if isinstance(allowed, list) and endpoint not in allowed:
        return None
    return f"{app_name}__{endpoint}"


def _extract_meta_tool_params(arguments: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(arguments, dict):
        raise RuntimeError("meta_tool_payload_invalid")
    if "params" in arguments and not isinstance(arguments.get("params"), dict):
        raise RuntimeError("meta_tool_params_invalid")
    if "endpoint_params" in arguments and not isinstance(arguments.get("endpoint_params"), dict):
        raise RuntimeError("meta_tool_params_invalid")
    params: Dict[str, Any] = {}
    endpoint_params = arguments.get("endpoint_params")
    if isinstance(endpoint_params, dict):
        params.update(endpoint_params)
    if isinstance(arguments.get("params"), dict):
        params.update(arguments["params"])
    flattened = {
        key: value
        for key, value in arguments.items()
        if key not in {"endpoint", "api_name", "params", "endpoint_params"}
    }
    for key, value in flattened.items():
        params.setdefault(key, value)
    return params


def _unwrap_tool_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    current = dict(arguments)
    for _ in range(3):
        wrapped = _extract_wrapped_args(current)
        if wrapped is None:
            break
        current = wrapped
    return current


def _extract_wrapped_args(arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in ("arguments", "input", "payload"):
        inner = arguments.get(key)
        if isinstance(inner, dict) and _has_known_arg_keys(inner):
            return inner
        if isinstance(inner, dict) and len(arguments) == 1:
            return inner
    return None


def _has_known_arg_keys(arguments: Dict[str, Any]) -> bool:
    known = {
        "command",
        "cmd",
        "path",
        "content",
        "pattern",
        "replacement",
        "timeout_s",
        "timeout",
        "count",
    }
    return any(key in arguments for key in known)


def _execute_builtin_tool(
    name: str,
    arguments: Dict[str, Any],
    sandbox: BaseSandbox,
    default_timeout_s: int,
) -> Optional[Any]:
    if name == "run_shell":
        command = _coerce_string(arguments, ("command", "cmd"))
        if not command:
            raise RuntimeError("command_missing")
        timeout = _coerce_timeout(arguments.get("timeout_s") or arguments.get("timeout"), default_timeout_s)
        return sandbox.exec(command, timeout=timeout)
    if name == "read_file":
        path = _coerce_string(arguments, ("path",))
        if not path:
            raise RuntimeError("path_missing")
        reader = getattr(sandbox, "read_file", None)
        if callable(reader):
            try:
                content = reader(path)
                return {"content": _decode_bytes(content)}
            except Exception:
                pass
        result = sandbox.exec(f"cat {shlex.quote(path)}", timeout=default_timeout_s)
        if getattr(result, "exit_code", 1) != 0:
            raise RuntimeError("read_file_failed")
        return {"content": str(getattr(result, "stdout", ""))}
    if name == "write_file":
        path = _coerce_string(arguments, ("path",))
        if not path:
            raise RuntimeError("path_missing")
        content = _coerce_string(arguments, ("content",))
        writer = getattr(sandbox, "write_file", None)
        if callable(writer):
            try:
                writer(path, content.encode("utf-8"))
                return {"status": "ok"}
            except Exception:
                pass
        command = _build_write_command(path, content)
        return sandbox.exec(command, timeout=default_timeout_s)
    if name == "replace_in_file":
        path = _coerce_string(arguments, ("path",))
        pattern = _coerce_string(arguments, ("pattern",))
        replacement = _coerce_string(arguments, ("replacement",))
        if not path or not pattern:
            raise RuntimeError("replace_in_file_missing_args")
        count = _coerce_optional_int(arguments.get("count"))
        command = _build_replace_command(path, pattern, replacement, count)
        return sandbox.exec(command, timeout=default_timeout_s)
    if name == "submit_patch_tool":
        timeout = _coerce_timeout(arguments.get("timeout_s") or arguments.get("timeout"), default_timeout_s)
        stage_untracked = _coerce_bool(_first_present(arguments, ("stage_untracked", "intent_to_add", "git_add")))
        if stage_untracked:
            _try_intent_to_add(sandbox, timeout)
        result = sandbox.exec("git diff", timeout=timeout)
        _write_submission_patch(sandbox, getattr(result, "stdout", ""))
        return result
    return None


def _coerce_timeout(value: Any, default: int) -> int:
    coerced = _coerce_optional_int(value)
    if coerced is None:
        return int(default)
    return max(1, coerced)


def _coerce_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(arguments: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for key in keys:
        if key in arguments:
            return arguments.get(key)
    return None


def _coerce_string(arguments: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    for key in keys:
        value = arguments.get(key)
        if value is not None:
            return str(value)
    return ""


def _decode_bytes(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _resolve_final_answer(output: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    if not metadata:
        return ""
    spec = metadata.get("final_answer_from") or metadata.get("final_answer_key")
    if not spec:
        return ""
    keys: list[str] = []
    if isinstance(spec, str):
        keys = [spec]
    elif isinstance(spec, list):
        keys = [str(item) for item in spec if item is not None]
    for key in keys:
        value = _extract_output_value(output, key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_output_value(output: Dict[str, Any], key: str) -> Any:
    if key in output:
        return output.get(key)
    nested = output.get("output")
    if isinstance(nested, dict):
        return nested.get(key)
    return None


def _build_write_command(path: str, content: str) -> str:
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    return (
        "python - <<'PY'\n"
        "import base64\n"
        "from pathlib import Path\n"
        f"path = {path!r}\n"
        f"data = base64.b64decode({b64!r})\n"
        "target = Path(path)\n"
        "target.parent.mkdir(parents=True, exist_ok=True)\n"
        "target.write_bytes(data)\n"
        "PY\n"
    )


def _build_replace_command(path: str, pattern: str, replacement: str, count: Optional[int]) -> str:
    replace_count = 0 if count is None else max(0, int(count))
    return (
        "python - <<'PY'\n"
        "import re\n"
        "from pathlib import Path\n"
        f"path = {path!r}\n"
        f"pattern = {pattern!r}\n"
        f"replacement = {replacement!r}\n"
        f"count = {replace_count}\n"
        "target = Path(path)\n"
        "text = target.read_text(encoding='utf-8', errors='replace')\n"
        "updated, _ = re.subn(pattern, replacement, text, count=count)\n"
        "target.write_text(updated, encoding='utf-8')\n"
        "PY\n"
    )


def _try_intent_to_add(sandbox: BaseSandbox, timeout: int) -> None:
    try:
        sandbox.exec("git add -N -- .", timeout=timeout)
    except Exception:
        return


def _write_submission_patch(sandbox: BaseSandbox, content: Any) -> None:
    if not content:
        return
    payload = str(content)
    writer = getattr(sandbox, "write_file", None)
    if callable(writer):
        try:
            writer("/workspace/submission.patch", payload.encode("utf-8"))
            return
        except Exception:
            pass
    try:
        command = _build_write_command("/workspace/submission.patch", payload)
        sandbox.exec(command, timeout=30)
    except Exception:
        return
