"""Toolchain adapter responsible for tool schema preparation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState
from gage_eval.role.adapters.tool_docs import (
    build_app_catalog,
    build_meta_tools,
    build_tool_documentation,
)
from gage_eval.mcp import McpClient
from gage_eval.sandbox.provider import SandboxProvider


@registry.asset(
    "roles",
    "toolchain",
    desc="Unified role adapter for external tools/MCP",
    tags=("role", "tool"),
    role_type="toolchain",
)
class ToolchainAdapter(RoleAdapter):
    def __init__(
        self,
        adapter_id: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        mcp_client_id: Optional[str] = None,
        mcp_client: Optional[McpClient] = None,
        **params,
    ) -> None:
        super().__init__(
            adapter_id=adapter_id,
            role_type="toolchain",
            capabilities=tuple(tool.get("name") for tool in tools or [] if isinstance(tool, dict)),
            resource_requirement=params.pop("resource_requirement", None),
            sandbox_config=params.pop("sandbox_config", None),
        )
        self._tools = list(tools or [])
        self._mcp_client_id = mcp_client_id or (mcp_client.mcp_client_id if mcp_client else None)
        self._mcp_client = mcp_client
        self._tool_allowlist = _normalize_tool_filter(params.pop("tool_allowlist", None))
        self._tool_prefixes = _normalize_tool_filter(params.pop("tool_prefixes", None))
        self._max_tools = _coerce_int(params.pop("max_tools", None))
        self._meta_tool_mode = bool(params.pop("meta_tool_mode", False))
        self._tool_doc_enabled = bool(params.pop("tool_doc_enabled", False))
        self._tool_doc_format = params.pop("tool_doc_format", "text")
        self._tool_doc_max_endpoints = _coerce_int(params.pop("tool_doc_max_endpoints", None))
        self._tool_doc_max_chars = _coerce_int(params.pop("tool_doc_max_chars", None))
        self._tool_doc_allowed_apps = _normalize_tool_filter(params.pop("tool_doc_allowed_apps", None))
        if self._mcp_client_id and self._mcp_client is None:
            logger.warning(
                "ToolchainAdapter '{}' configured with mcp_client_id='{}' but no McpClient instance was injected.",
                adapter_id,
                self._mcp_client_id,
            )

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        """Assemble tool schema for downstream agent execution.

        Args:
            payload: Adapter invocation payload.
            state: Per-sample adapter state.

        Returns:
            Tool schema payload with optional MCP metadata.
        """

        sample = payload.get("sample", {}) if isinstance(payload, dict) else {}
        sandbox_provider = payload.get("sandbox_provider") if isinstance(payload, dict) else None
        mcp_tools = []
        raw_mcp_tools: List[Dict[str, Any]] = []
        if self._mcp_client and isinstance(sandbox_provider, SandboxProvider):
            handle = sandbox_provider.get_handle()
            runtime_handle = handle.runtime_handle if handle else {}
            _sync_mcp_endpoint(self._mcp_client, runtime_handle)
        if self._mcp_client:
            raw_mcp_tools = list(self._mcp_client.list_tools())
        if self._meta_tool_mode and raw_mcp_tools:
            allowed_apps = _resolve_allowed_apps(sample, self._tool_doc_allowed_apps)
            catalog = build_app_catalog(raw_mcp_tools, allowed_apps=allowed_apps)
            documentation = build_tool_documentation(
                catalog,
                doc_format=self._tool_doc_format,
                max_endpoints=self._tool_doc_max_endpoints,
                max_chars=self._tool_doc_max_chars,
            )
            meta_tools = build_meta_tools(
                documentation.endpoints_by_app,
                mcp_client_id=self._mcp_client_id,
            )
            tools_schema = _merge_tools(self._tools, sample.get("tools") or [])
            tools_schema = _merge_tools(tools_schema, meta_tools)
            return {
                "tools_schema": tools_schema,
                "mcp_client_id": self._mcp_client_id,
                "tool_documentation": documentation.text,
                "tool_documentation_meta": documentation.meta,
            }
        if raw_mcp_tools:
            mcp_tools = _map_mcp_tools(raw_mcp_tools, self._mcp_client_id)
        if self._tool_allowlist or self._tool_prefixes or self._max_tools:
            mcp_tools = _filter_tools(
                mcp_tools,
                allowlist=self._tool_allowlist,
                prefixes=self._tool_prefixes,
                max_tools=self._max_tools,
            )
        tool_docs_payload: Dict[str, Any] = {}
        if raw_mcp_tools and mcp_tools and self._tool_doc_enabled:
            allowed_apps = _resolve_allowed_apps(sample, self._tool_doc_allowed_apps)
            doc_tools = _select_raw_tools(raw_mcp_tools, mcp_tools)
            catalog = build_app_catalog(doc_tools, allowed_apps=allowed_apps)
            documentation = build_tool_documentation(
                catalog,
                doc_format=self._tool_doc_format,
                max_endpoints=self._tool_doc_max_endpoints,
                max_chars=self._tool_doc_max_chars,
            )
            tool_docs_payload = {
                "tool_documentation": documentation.text,
                "tool_documentation_meta": documentation.meta,
            }
        tools_schema = _merge_tools(self._tools, sample.get("tools") or [])
        tools_schema = _merge_tools(tools_schema, mcp_tools)
        payload = {
            "tools_schema": tools_schema,
            "mcp_client_id": self._mcp_client_id,
        }
        payload.update(tool_docs_payload)
        return payload


def _sync_mcp_endpoint(mcp_client: McpClient, runtime_handle: Dict[str, Any]) -> None:
    endpoint = _extract_mcp_endpoint(runtime_handle)
    if not endpoint:
        return
    current = getattr(mcp_client, "endpoint", None)
    if current == endpoint:
        return
    disconnect = getattr(mcp_client, "disconnect", None)
    if callable(disconnect):
        disconnect()
    setattr(mcp_client, "endpoint", endpoint)


def _extract_mcp_endpoint(runtime_handle: Dict[str, Any]) -> Optional[str]:
    for key in ("mcp_endpoint", "mcp_url", "remote_mcp_url"):
        value = runtime_handle.get(key)
        if value:
            return str(value).rstrip("/")
    return None


def _map_mcp_tools(raw_tools: Sequence[Dict[str, Any]], mcp_client_id: Optional[str]) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for tool in raw_tools:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description") or "",
                    "parameters": _strip_nulls(
                        tool.get("inputSchema") or tool.get("parameters") or {}
                    ),
                },
                "x-gage": {"mcp_client_id": mcp_client_id},
            }
        )
    return tools


def _resolve_allowed_apps(sample: Dict[str, Any], fallback: Sequence[str]) -> List[str]:
    if not isinstance(sample, dict):
        return list(fallback)
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    appworld = metadata.get("appworld") if isinstance(metadata.get("appworld"), dict) else {}
    allowed = appworld.get("allowed_apps")
    if isinstance(allowed, list) and allowed:
        return [str(item) for item in allowed if item]
    return list(fallback)


def _merge_tools(default_tools: List[Dict[str, Any]], sample_tools: Any) -> List[Dict[str, Any]]:
    merged = []
    merged = _dedupe_tools(merged, default_tools)
    merged = _dedupe_tools(merged, sample_tools)
    return merged


def _dedupe_tools(existing: List[Dict[str, Any]], tools: Any) -> List[Dict[str, Any]]:
    merged = list(existing)
    if isinstance(tools, dict):
        tools = [tools]
    for tool in tools or []:
        normalized = _normalize_tool_entry(tool)
        if not normalized:
            continue
        name = normalized.get("function", {}).get("name") if normalized.get("type") == "function" else None
        if name:
            merged = [item for item in merged if item.get("function", {}).get("name") != name]
        merged.append(normalized)
    return merged


def _normalize_tool_entry(tool: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(tool, dict):
        return None
    if tool.get("type") == "function" and "function" in tool:
        normalized = dict(tool)
        function = normalized.get("function")
        if isinstance(function, dict):
            function = dict(function)
            function["description"] = function.get("description") or ""
            if "parameters" in function:
                function["parameters"] = _strip_nulls(function.get("parameters") or {})
            normalized["function"] = function
        return normalized
    if "name" in tool and "parameters" in tool:
        return {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description") or "",
                "parameters": _strip_nulls(tool.get("parameters") or {}),
            },
        }
    return dict(tool)


def _strip_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            cleaned[key] = _strip_nulls(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_nulls(item) for item in value if item is not None]
    return value


def _normalize_tool_filter(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item]


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _filter_tools(
    tools: List[Dict[str, Any]],
    *,
    allowlist: List[str],
    prefixes: List[str],
    max_tools: Optional[int],
) -> List[Dict[str, Any]]:
    if not allowlist and not prefixes and max_tools is None:
        return tools
    filtered: List[Dict[str, Any]] = []
    for tool in tools:
        name = _extract_tool_name(tool)
        if name is None:
            continue
        if allowlist or prefixes:
            if allowlist and name in allowlist:
                filtered.append(tool)
                continue
            if prefixes and _matches_prefix(name, prefixes):
                filtered.append(tool)
            continue
        filtered.append(tool)
    if max_tools is not None:
        return filtered[: max(0, max_tools)]
    return filtered


def _extract_tool_name(tool: Dict[str, Any]) -> Optional[str]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return tool["function"].get("name")
    return tool.get("name")


def _matches_prefix(name: str, prefixes: List[str]) -> bool:
    for prefix in prefixes:
        if name == prefix:
            return True
        if name.startswith(prefix + "__"):
            return True
        if name.startswith(prefix + "_"):
            return True
    return False


def _select_raw_tools(
    raw_tools: Sequence[Dict[str, Any]],
    filtered_tools: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    allowed_names = {
        name for name in (_extract_tool_name(tool) for tool in filtered_tools) if isinstance(name, str)
    }
    if not allowed_names:
        return []
    return [tool for tool in raw_tools if tool.get("name") in allowed_names]
