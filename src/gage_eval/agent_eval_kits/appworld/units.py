from __future__ import annotations

from typing import Any

from gage_eval.agent_eval_kits.common import extract_instruction, normalize_tools
from gage_eval.evaluation.support_artifacts import resolve_support_tools

_APPWORLD_TOOL_USE_CONTRACT = (
    "You are operating inside AppWorld. Use the provided AppWorld tools and endpoints to "
    "inspect the allowed apps and determine the answer. Do not answer from prior knowledge "
    "or refuse access when tools are available."
)


def build_appworld_prompt_context(
    sample: dict[str, object],
    runtime_handle: dict[str, object],
    initialize_result: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build AppWorld prompt/runtime context from sample metadata and endpoints."""

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    appworld = metadata.get("appworld") if isinstance(metadata.get("appworld"), dict) else {}
    initialize_payload = initialize_result if isinstance(initialize_result, dict) else {}
    instruction = initialize_payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = extract_instruction(sample)
    return {
        "instruction": instruction,
        "allowed_apps": list(appworld.get("allowed_apps") or []),
        "task_id": appworld.get("task_id"),
        "ground_truth_mode": appworld.get("ground_truth_mode"),
        "env_endpoint": runtime_handle.get("env_endpoint"),
        "apis_endpoint": runtime_handle.get("apis_endpoint"),
        "mcp_endpoint": runtime_handle.get("mcp_endpoint"),
    }


def build_appworld_messages(
    sample: dict[str, object],
    *,
    instruction_override: str | None = None,
) -> list[dict[str, object]]:
    """Build framework-loop messages for AppWorld."""

    instruction = ""
    if isinstance(instruction_override, str) and instruction_override.strip():
        instruction = instruction_override.strip()
    else:
        instruction = extract_instruction(sample)
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": _APPWORLD_TOOL_USE_CONTRACT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": instruction}],
        },
    ]


def build_appworld_tools(sample: dict[str, object]) -> list[dict[str, object]]:
    """Build AppWorld tool schemas."""

    tools = resolve_support_tools(sample)
    if tools:
        return tools
    return normalize_tools(sample)


def fetch_mcp_tool_schemas(
    mcp_endpoint: str,
    mcp_client_id: str | None,
    *,
    allowed_apps: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch live tool schemas from the AppWorld MCP server.

    Uses AppWorldStreamableMcpClient to connect to the running sandbox MCP
    endpoint, lists tools, filters by allowed_apps, and maps each into the
    OpenAI function-call format expected by the ToolRouter.  The client is
    disconnected immediately after tool discovery so no persistent session
    is kept in the kit layer.

    The returned schemas carry ``x-gage: {mcp_client_id: <id>}`` so the
    ToolRouter can dispatch tool calls through the already-registered MCP
    client (e.g. ``appworld_env`` from the pipeline config).
    """

    from gage_eval.sandbox.integrations.appworld.mcp_client import AppWorldStreamableMcpClient

    client = AppWorldStreamableMcpClient(
        mcp_client_id=mcp_client_id or "appworld",
        endpoint=mcp_endpoint,
        params={"session_retry_attempts": 1, "session_retry_timeout_s": 30},
    )
    try:
        raw_tools = client.list_tools()
    finally:
        client.disconnect()

    allowed = set(allowed_apps) if allowed_apps else set()
    result: list[dict[str, Any]] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name") or ""
        if allowed:
            app = name.split("__", 1)[0] if "__" in name else name
            if app not in allowed:
                continue
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description") or "",
                "parameters": _strip_nulls(
                    tool.get("inputSchema") or tool.get("parameters") or {}
                ),
            },
        }
        if mcp_client_id:
            schema["x-gage"] = {"mcp_client_id": mcp_client_id}
        result.append(schema)
    return result


def _strip_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_nulls(item) for item in value if item is not None]
    return value


def build_appworld_instruction(
    sample: dict[str, object],
    *,
    instruction_override: str | None = None,
) -> str:
    """Build the AppWorld instruction with the tool-use contract."""

    if isinstance(instruction_override, str) and instruction_override.strip():
        instruction = instruction_override.strip()
    else:
        instruction = extract_instruction(sample)
    if not instruction:
        return _APPWORLD_TOOL_USE_CONTRACT
    if _APPWORLD_TOOL_USE_CONTRACT in instruction:
        return instruction
    return f"{instruction.rstrip()}\n\nTool-use contract: {_APPWORLD_TOOL_USE_CONTRACT}"
