"""Lightweight tool router for agent tool calls."""

from __future__ import annotations

import json
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
        payload = json.dumps({"tool": name, "arguments": arguments}, ensure_ascii=True)
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
