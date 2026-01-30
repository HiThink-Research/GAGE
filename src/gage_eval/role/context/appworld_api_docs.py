"""AppWorld ApiDocs context provider."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from gage_eval.registry import registry


_API_DOCS_TOOL_PREFIX = "api_docs__"
_DEFAULT_PAGE_LIMIT = 5
_DEFAULT_MAX_PAGES = 3
_DEFAULT_MAX_CHARS = 8000
_DEFAULT_MODE = "search"
_CACHE_KEY = "appworld_api_docs_cache"


@registry.asset(
    "context_impls",
    "appworld_api_docs",
    desc="AppWorld ApiDocs context provider",
    tags=("appworld", "context"),
)
class AppWorldApiDocsContext:
    """Provide ApiDocs context via MCP tool calls."""

    def __init__(
        self,
        *,
        mcp_client_id: Optional[str] = None,
        mcp_client: Optional[Any] = None,
        page_limit: int = _DEFAULT_PAGE_LIMIT,
        max_pages: int = _DEFAULT_MAX_PAGES,
        max_chars: int = _DEFAULT_MAX_CHARS,
        include_app_descriptions: bool = False,
        mode: str = _DEFAULT_MODE,
        auto_discover_apps: bool = False,
        max_apps: Optional[int] = None,
        max_apis_per_app: Optional[int] = None,
        exclude_apps: Optional[Iterable[str]] = None,
    ) -> None:
        self._mcp_client_id = mcp_client_id
        self._mcp_client = mcp_client
        self._page_limit = max(1, int(page_limit))
        self._max_pages = max(1, int(max_pages))
        self._max_chars = max(100, int(max_chars))
        self._include_app_descriptions = bool(include_app_descriptions)
        self._mode = str(mode or _DEFAULT_MODE)
        self._auto_discover_apps = bool(auto_discover_apps)
        self._max_apps = _coerce_int(max_apps)
        self._max_apis_per_app = _coerce_int(max_apis_per_app)
        self._exclude_apps = {str(item) for item in (exclude_apps or []) if item}

    def provide(self, payload: Dict[str, Any], state=None) -> Dict[str, Any]:
        """Fetch ApiDocs context using MCP calls.

        Args:
            payload: Adapter payload containing sample and params.
            state: RoleAdapterState for per-sample cache.

        Returns:
            A dict with api_docs_context and optional observability events.
        """

        sample = payload.get("sample") or {}
        params = payload.get("params") or {}
        mcp_client = self._resolve_client(params)
        if mcp_client is None:
            raise RuntimeError("appworld_api_docs_missing_mcp_client")

        cache = _resolve_cache(state)
        events: List[Dict[str, Any]] = []
        chunks: List[str] = []

        mode = str(params.get("mode") or self._mode)
        if mode == "api_descriptions":
            params.setdefault("auto_discover_apps", self._auto_discover_apps)
            return _build_api_descriptions_context(
                mcp_client,
                params,
                cache,
                events,
                max_chars=self._max_chars,
                include_app_descriptions=params.get("include_app_descriptions", self._include_app_descriptions),
                max_apps=_coerce_int(params.get("max_apps")) or self._max_apps,
                max_apis_per_app=_coerce_int(params.get("max_apis_per_app")) or self._max_apis_per_app,
                exclude_apps=_normalize_list(params.get("exclude_apps")) or list(self._exclude_apps),
            )

        if params.get("include_app_descriptions", self._include_app_descriptions):
            response = _call_api_docs_tool(
                mcp_client,
                "show_app_descriptions",
                {},
                cache,
                events,
            )
            chunks.append(_format_api_docs_payload("app_descriptions", response))

        app_names = _normalize_list(params.get("app_names"))
        api_names = _normalize_api_names(params.get("api_names"))
        if app_names:
            for app_name in app_names:
                response = _call_api_docs_tool(
                    mcp_client,
                    "show_api_descriptions",
                    {"app_name": app_name},
                    cache,
                    events,
                )
                chunks.append(_format_api_docs_payload(f"{app_name}: api_descriptions", response))

        if api_names:
            for app_name, names in api_names:
                for api_name in names:
                    response = _call_api_docs_tool(
                        mcp_client,
                        "show_api_doc",
                        {"app_name": app_name, "api_name": api_name},
                        cache,
                        events,
                    )
                    chunks.append(_format_api_docs_payload(f"{app_name}.{api_name}", response))
        else:
            query = params.get("query") or _extract_instruction(sample)
            if isinstance(query, str) and query.strip():
                for page_index in range(self._max_pages):
                    response = _call_api_docs_tool(
                        mcp_client,
                        "search_api_docs",
                        {
                            "query": query,
                            "page_index": page_index,
                            "page_limit": self._page_limit,
                        },
                        cache,
                        events,
                    )
                    if not response:
                        break
                    chunks.append(_format_api_docs_payload(f"search_page_{page_index}", response))

        context = _truncate_text("\n\n".join(chunk for chunk in chunks if chunk), self._max_chars)
        result: Dict[str, Any] = {
            "api_docs_context": context,
        }
        if events:
            result["observability_events"] = events
        return result

    def _resolve_client(self, params: Dict[str, Any]) -> Optional[Any]:
        if self._mcp_client is not None:
            return self._mcp_client
        return params.get("mcp_client")


def _resolve_cache(state: Any) -> Dict[Tuple[str, str, str, int], Any]:
    if state is None:
        return {}
    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, dict):
        return {}
    return metadata.setdefault(_CACHE_KEY, {})


def _call_api_docs_tool(
    mcp_client: Any,
    api_name: str,
    params: Dict[str, Any],
    cache: Dict[Tuple[str, str, str, int], Any],
    events: List[Dict[str, Any]],
) -> Any:
    cache_key = _build_cache_key(api_name, params)
    if cache_key in cache:
        events.append(_build_event(api_name, params, cache_hit=True))
        return cache[cache_key]
    tool_name = f"{_API_DOCS_TOOL_PREFIX}{api_name}"
    response = mcp_client.call_tool(tool_name, params)
    cache[cache_key] = response
    events.append(_build_event(api_name, params, cache_hit=False))
    return response


def _build_cache_key(api_name: str, params: Dict[str, Any]) -> Tuple[str, str, str, int]:
    app_name = str(params.get("app_name") or "")
    query = str(params.get("query") or "")
    page_index = int(params.get("page_index") or 0)
    key = f"{app_name}:{params.get('api_name') or ''}:{query}"
    return api_name, app_name, key, page_index


def _build_event(api_name: str, params: Dict[str, Any], *, cache_hit: bool) -> Dict[str, Any]:
    payload = {
        "api_name": api_name,
        "app_name": params.get("app_name"),
        "query": params.get("query"),
        "page_index": params.get("page_index"),
        "cache_hit": cache_hit,
    }
    return {"event": "api_docs_query", "payload": payload}


def _format_api_docs_payload(title: str, payload: Any) -> str:
    if payload is None:
        return ""
    try:
        rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    except TypeError:
        rendered = json.dumps(_stringify_keys(payload), indent=2, ensure_ascii=True)
    return f"# {title}\n{rendered}"


def _stringify_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stringify_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_keys(item) for item in value]
    return value


def _extract_instruction(sample: Dict[str, Any]) -> str:
    instruction = sample.get("instruction")
    if isinstance(instruction, str):
        return instruction
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0] if isinstance(messages[0], dict) else None
        if first:
            content = first.get("content")
            if isinstance(content, list) and content:
                text = content[0].get("text")
                if isinstance(text, str):
                    return text
            if isinstance(content, str):
                return content
    return ""


def _normalize_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value if item]
    return []


def _normalize_api_names(value: Any) -> List[Tuple[str, List[str]]]:
    if not value:
        return []
    if isinstance(value, dict):
        return [(str(app), _normalize_list(names)) for app, names in value.items() if app]
    return []


def _truncate_text(text: str, max_len: int) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def _build_api_descriptions_context(
    mcp_client: Any,
    params: Dict[str, Any],
    cache: Dict[Tuple[str, str, str, int], Any],
    events: List[Dict[str, Any]],
    *,
    max_chars: int,
    include_app_descriptions: bool,
    max_apps: Optional[int],
    max_apis_per_app: Optional[int],
    exclude_apps: List[str],
) -> Dict[str, Any]:
    tools = _list_mcp_tools(mcp_client, cache, events)
    app_descriptions = _build_app_descriptions_from_tools(tools, exclude_apps)
    app_descriptions_string = _format_app_descriptions_yaml(app_descriptions)
    app_names = _resolve_app_names(params, app_descriptions, exclude_apps)
    if max_apps is not None:
        app_names = app_names[: max_apps]

    api_descriptions = _build_api_descriptions_from_tools(
        tools,
        app_names=app_names,
        exclude_apps=exclude_apps,
        max_apis_per_app=max_apis_per_app,
    )
    allowed_apis, allowed_tools = _build_allowed_api_sets(tools)

    context = _format_api_descriptions(
        app_descriptions if include_app_descriptions else None,
        api_descriptions,
    )
    context = _truncate_text(context, max_chars)
    result: Dict[str, Any] = {
        "api_descriptions_context": context,
        "api_descriptions_allowed_apis": allowed_apis,
        "api_descriptions_allowed_tools": allowed_tools,
    }
    if app_descriptions_string:
        result["app_descriptions_string"] = app_descriptions_string
    if events:
        result["observability_events"] = events
    return result


def _resolve_app_names(
    params: Dict[str, Any],
    app_descriptions: List[Dict[str, Any]],
    exclude_apps: List[str],
) -> List[str]:
    app_names = _normalize_list(params.get("app_names"))
    auto_discover = bool(params.get("auto_discover_apps"))
    if not app_names or auto_discover or "*" in app_names or "all" in app_names:
        app_names = [
            str(item.get("name"))
            for item in app_descriptions
            if isinstance(item, dict) and item.get("name")
        ]
    excluded = {str(item) for item in exclude_apps if item}
    return [name for name in app_names if name and name not in excluded]


def _format_api_descriptions(
    app_descriptions: Optional[Any],
    api_descriptions: Dict[str, List[Dict[str, Any]]],
) -> str:
    lines: List[str] = []
    if isinstance(app_descriptions, list):
        lines.append("app_descriptions:")
        for item in app_descriptions:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            desc = str(item.get("description") or "").strip()
            if not name:
                continue
            lines.append(f"  - name: {name}")
            if desc:
                lines.append(f"    description: {desc}")
    if api_descriptions:
        lines.append("api_descriptions:")
        for app_name in sorted(api_descriptions.keys()):
            lines.append(f"  {app_name}:")
            for api in api_descriptions.get(app_name, []):
                if not isinstance(api, dict):
                    continue
                name = str(api.get("name") or "").strip()
                desc = str(api.get("description") or "").strip()
                if not name:
                    continue
                lines.append(f"    - name: {name}")
                if desc:
                    lines.append(f"      description: {desc}")
    return "\n".join(lines).strip()


def _format_app_descriptions_yaml(app_descriptions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in app_descriptions:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        desc = str(item.get("description") or "").strip()
        if desc:
            lines.append(f"{name}: {desc}")
        else:
            lines.append(f'{name}: ""')
    return "\n".join(lines).strip()


def _list_mcp_tools(
    mcp_client: Any,
    cache: Dict[Tuple[str, str, str, int], Any],
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cache_key = ("list_tools", "", "", 0)
    if cache_key in cache:
        events.append(_build_event("list_tools", {}, cache_hit=True))
        cached = cache[cache_key]
        return cached if isinstance(cached, list) else []
    tools = list(mcp_client.list_tools())
    cache[cache_key] = tools
    events.append(_build_event("list_tools", {}, cache_hit=False))
    return [tool for tool in tools if isinstance(tool, dict)]


def _build_app_descriptions_from_tools(
    tools: List[Dict[str, Any]],
    exclude_apps: List[str],
) -> List[Dict[str, Any]]:
    apps: Dict[str, Dict[str, str]] = {}
    excluded = {str(item) for item in exclude_apps if item}
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str):
            continue
        app_endpoint = _split_app_endpoint(name)
        if app_endpoint is None:
            continue
        app, _ = app_endpoint
        if app in excluded:
            continue
        apps.setdefault(app, {"name": app, "description": ""})
    return [apps[app] for app in sorted(apps.keys())]


def _build_api_descriptions_from_tools(
    tools: List[Dict[str, Any]],
    *,
    app_names: List[str],
    exclude_apps: List[str],
    max_apis_per_app: Optional[int],
) -> Dict[str, List[Dict[str, Any]]]:
    allowed_apps = set(app_names)
    excluded = {str(item) for item in exclude_apps if item}
    api_descriptions: Dict[str, List[Dict[str, Any]]] = {app: [] for app in app_names}
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str):
            continue
        app_endpoint = _split_app_endpoint(name)
        if app_endpoint is None:
            continue
        app, endpoint = app_endpoint
        if app in excluded or (allowed_apps and app not in allowed_apps):
            continue
        api_descriptions.setdefault(app, []).append(
            {
                "name": endpoint,
                "description": str(tool.get("description") or ""),
            }
        )
    for app, apis in list(api_descriptions.items()):
        apis.sort(key=lambda item: str(item.get("name") or ""))
        if max_apis_per_app is not None:
            api_descriptions[app] = apis[:max_apis_per_app]
        else:
            api_descriptions[app] = apis
    return api_descriptions


def _split_app_endpoint(name: str) -> Optional[Tuple[str, str]]:
    if "__" not in name:
        return None
    app, endpoint = name.split("__", 1)
    if not app or not endpoint:
        return None
    return app, endpoint


def _build_allowed_api_sets(tools: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    api_names: List[str] = []
    tool_names: List[str] = []
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str):
            continue
        app_endpoint = _split_app_endpoint(name)
        if app_endpoint is None:
            continue
        app, endpoint = app_endpoint
        api_names.append(f"{app}.{endpoint}".lower())
        tool_names.append(name)
    return _dedupe_items(api_names), _dedupe_items(tool_names)


def _dedupe_items(items: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None
