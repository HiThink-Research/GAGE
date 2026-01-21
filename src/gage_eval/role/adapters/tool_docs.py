"""Meta-tool documentation builders for large tool inventories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_DEFAULT_RESULT_TYPE = "result"
_TYPE_MAP = {
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "string": "string",
    "array": "list",
    "object": "object",
}


@dataclass(frozen=True)
class ToolEndpoint:
    """Represents a single AppWorld endpoint used for documentation."""

    app: str
    endpoint: str
    description: str
    input_schema: Dict[str, Any]


@dataclass(frozen=True)
class ToolDocumentation:
    """Aggregated tool documentation and metadata."""

    text: str
    meta: Dict[str, Any]
    endpoints_by_app: Dict[str, List[str]]


def build_app_catalog(
    tools: Sequence[Dict[str, Any]],
    *,
    allowed_apps: Optional[Sequence[str]] = None,
) -> Dict[str, List[ToolEndpoint]]:
    """Build an app-to-endpoint catalog from MCP tools.

    Args:
        tools: MCP tool definitions as returned by list_tools.
        allowed_apps: Optional allowlist of app names.

    Returns:
        Mapping of app name to ToolEndpoint definitions.
    """

    allowed = _normalize_allowed_apps(allowed_apps)
    catalog: Dict[str, List[ToolEndpoint]] = {}
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str):
            continue
        app_endpoint = _split_app_endpoint(name)
        if app_endpoint is None:
            continue
        app, endpoint = app_endpoint
        if allowed and app not in allowed:
            continue
        input_schema = tool.get("inputSchema") or tool.get("parameters") or {}
        catalog.setdefault(app, []).append(
            ToolEndpoint(
                app=app,
                endpoint=endpoint,
                description=str(tool.get("description") or ""),
                input_schema=dict(input_schema) if isinstance(input_schema, dict) else {},
            )
        )
    for app, endpoints in catalog.items():
        catalog[app] = sorted(endpoints, key=lambda entry: entry.endpoint)
    return catalog


def build_tool_documentation(
    catalog: Dict[str, List[ToolEndpoint]],
    *,
    doc_format: str = "text",
    max_endpoints: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> ToolDocumentation:
    """Render a compact tool documentation payload.

    Args:
        catalog: App-to-endpoint mapping.
        doc_format: Output format identifier.
        max_endpoints: Max endpoints per app.
        max_chars: Max characters in the output documentation.

    Returns:
        ToolDocumentation bundle.
    """

    max_chars = _coerce_positive(max_chars)
    max_endpoints = _coerce_positive(max_endpoints)
    if doc_format == "text":
        return _build_text_documentation(catalog, max_endpoints=max_endpoints, max_chars=max_chars)
    if doc_format == "app_kv":
        return _build_app_kv_documentation(catalog, max_endpoints=max_endpoints, max_chars=max_chars)
    raise ValueError(f"Unsupported tool_doc_format '{doc_format}'")


def _build_text_documentation(
    catalog: Dict[str, List[ToolEndpoint]],
    *,
    max_endpoints: Optional[int],
    max_chars: Optional[int],
) -> ToolDocumentation:
    lines: List[str] = []
    endpoints_by_app: Dict[str, List[str]] = {}
    total_endpoints = 0
    current_chars = 0
    truncated = False
    for app in _ordered_apps(catalog.keys()):
        endpoints = catalog.get(app) or []
        if max_endpoints is not None:
            endpoints = endpoints[:max_endpoints]
        if not endpoints:
            continue
        if not _append_line(lines, f"[{app}]", max_chars, current_chars):
            truncated = True
            break
        current_chars = _count_chars(lines)
        endpoints_by_app[app] = []
        for endpoint in endpoints:
            signature = _build_signature(endpoint)
            if not _append_line(lines, signature, max_chars, current_chars):
                truncated = True
                break
            current_chars = _count_chars(lines)
            endpoints_by_app[app].append(endpoint.endpoint)
            total_endpoints += 1
            for note in _build_notes(endpoint, indent="  "):
                if not _append_line(lines, note, max_chars, current_chars):
                    truncated = True
                    break
                current_chars = _count_chars(lines)
            if truncated:
                break
        if truncated:
            break
        _append_line(lines, "", max_chars, current_chars)
        current_chars = _count_chars(lines)
    return _finalize_documentation(
        lines,
        endpoints_by_app=endpoints_by_app,
        total_endpoints=total_endpoints,
        truncated=truncated,
        doc_format="text",
        max_endpoints=max_endpoints,
        max_chars=max_chars,
    )


def _build_app_kv_documentation(
    catalog: Dict[str, List[ToolEndpoint]],
    *,
    max_endpoints: Optional[int],
    max_chars: Optional[int],
) -> ToolDocumentation:
    lines: List[str] = []
    endpoints_by_app: Dict[str, List[str]] = {}
    total_endpoints = 0
    current_chars = 0
    truncated = False
    for app in _ordered_apps(catalog.keys()):
        endpoints = catalog.get(app) or []
        if max_endpoints is not None:
            endpoints = endpoints[:max_endpoints]
        if not endpoints:
            continue
        if not _append_line(lines, f"{app}:", max_chars, current_chars):
            truncated = True
            break
        current_chars = _count_chars(lines)
        endpoints_by_app[app] = []
        for endpoint in endpoints:
            description = endpoint.description.strip() if endpoint.description else ""
            if not description:
                description = f"Call {endpoint.app} {endpoint.endpoint}."
            line = f"  {endpoint.endpoint}: {description}"
            if not _append_line(lines, line, max_chars, current_chars):
                truncated = True
                break
            current_chars = _count_chars(lines)
            endpoints_by_app[app].append(endpoint.endpoint)
            total_endpoints += 1
            for note in _build_notes(endpoint, indent="    "):
                if not _append_line(lines, note, max_chars, current_chars):
                    truncated = True
                    break
                current_chars = _count_chars(lines)
            if truncated:
                break
        if truncated:
            break
        _append_line(lines, "", max_chars, current_chars)
        current_chars = _count_chars(lines)
    return _finalize_documentation(
        lines,
        endpoints_by_app=endpoints_by_app,
        total_endpoints=total_endpoints,
        truncated=truncated,
        doc_format="app_kv",
        max_endpoints=max_endpoints,
        max_chars=max_chars,
    )


def build_meta_tools(
    endpoints_by_app: Dict[str, List[str]],
    *,
    mcp_client_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate Meta-Tools based on the endpoints catalog."""

    tools: List[Dict[str, Any]] = []
    for app in _ordered_apps(endpoints_by_app.keys()):
        endpoints = endpoints_by_app.get(app) or []
        if not endpoints:
            continue
        name = f"call_{app}"
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Call {app} endpoints in AppWorld.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "endpoint": {"type": "string"},
                            "params": {"type": "object"},
                        },
                        "required": ["endpoint"],
                    },
                },
                "x-gage": {
                    "meta_tool": True,
                    "app_name": app,
                    "allowed_endpoints": list(endpoints),
                    "mcp_client_id": mcp_client_id,
                },
            }
        )
    return tools


def _split_app_endpoint(name: str) -> Optional[Tuple[str, str]]:
    if "__" not in name:
        return None
    app, endpoint = name.split("__", 1)
    if not app or not endpoint:
        return None
    return app, endpoint


def _ordered_apps(apps: Iterable[str]) -> List[str]:
    ordered = sorted(str(app) for app in apps if app)
    if "supervisor" in ordered:
        ordered.remove("supervisor")
        ordered.insert(0, "supervisor")
    return ordered


def _normalize_allowed_apps(value: Optional[Sequence[str]]) -> List[str]:
    if not value:
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _build_signature(endpoint: ToolEndpoint) -> str:
    schema = endpoint.input_schema if isinstance(endpoint.input_schema, dict) else {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    required = [str(item) for item in required]

    required_params = [name for name in properties.keys() if name in required]
    optional_params = [name for name in properties.keys() if name not in required]
    ordered = required_params + optional_params

    parts = []
    for name in ordered:
        info = properties.get(name) if isinstance(properties.get(name), dict) else {}
        param_type = _normalize_param_type(info)
        suffix = "" if name in required else "?"
        parts.append(f"{name}{suffix}: {param_type}")
    joined = ", ".join(parts)
    return f"{endpoint.app}.{endpoint.endpoint}({joined}) -> {_DEFAULT_RESULT_TYPE}"


def _normalize_param_type(schema: Dict[str, Any]) -> str:
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        raw_type = raw_type[0] if raw_type else None
    if isinstance(raw_type, str) and raw_type in _TYPE_MAP:
        return _TYPE_MAP[raw_type]
    if "enum" in schema:
        return "enum"
    if "anyOf" in schema or "oneOf" in schema:
        return "union"
    return "any"


def _build_notes(endpoint: ToolEndpoint, *, indent: str) -> List[str]:
    schema = endpoint.input_schema if isinstance(endpoint.input_schema, dict) else {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    notes: List[str] = []
    for name, info in properties.items():
        if not isinstance(info, dict):
            continue
        enum = info.get("enum")
        if isinstance(enum, list) and enum:
            if len(enum) <= 10:
                joined = ", ".join(str(item) for item in enum)
                notes.append(f"{indent}note: {name} in [{joined}]")
            else:
                notes.append(f"{indent}note: {name} has {len(enum)} enum values")
        minimum = info.get("minimum")
        if minimum == 0 and "page_index" in str(name):
            notes.append(f"{indent}note: page_index starts at 0")
    return notes


def _append_line(lines: List[str], line: str, max_chars: Optional[int], current_chars: int) -> bool:
    if max_chars is None:
        lines.append(line)
        return True
    projected = current_chars + len(line)
    if lines:
        projected += 1
    if projected > max_chars:
        return False
    lines.append(line)
    return True


def _count_chars(lines: List[str]) -> int:
    if not lines:
        return 0
    return len("\n".join(lines))


def _finalize_documentation(
    lines: List[str],
    *,
    endpoints_by_app: Dict[str, List[str]],
    total_endpoints: int,
    truncated: bool,
    doc_format: str,
    max_endpoints: Optional[int],
    max_chars: Optional[int],
) -> ToolDocumentation:
    text = "\n".join(lines).strip()
    meta = {
        "apps": len(endpoints_by_app),
        "endpoints": total_endpoints,
        "chars": len(text),
        "tokens_est": _estimate_tokens(text),
        "truncated": truncated,
        "doc_format": doc_format,
        "max_endpoints": max_endpoints,
        "max_chars": max_chars,
    }
    return ToolDocumentation(text=text, meta=meta, endpoints_by_app=endpoints_by_app)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _coerce_positive(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


__all__ = [
    "ToolDocumentation",
    "ToolEndpoint",
    "build_app_catalog",
    "build_meta_tools",
    "build_tool_documentation",
]
