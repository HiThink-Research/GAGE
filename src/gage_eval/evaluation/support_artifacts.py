"""Support artifact normalization helpers."""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from gage_eval.pipeline.step_contracts import get_step_adapter_id, get_step_type


SUPPORT_ARTIFACTS_KEY = "support_artifacts"
SUPPORT_OUTPUTS_KEY = "support_outputs"
SUPPORT_ARTIFACTS_VERSION = 1
DEFAULT_PROJECTION_MODE = "compact_latest"
VALID_PROJECTION_MODES = {"compact_latest", "legacy_full", "none"}

_LATEST_FIELD_NAMES = (
    "tool_documentation",
    "tool_documentation_meta",
    "api_descriptions_context",
    "api_descriptions_allowed_apis",
    "api_descriptions_allowed_tools",
    "answer",
)
_FILTER_LIST_FIELDS = (
    ("tool_allowlist", "allowlist"),
    ("tool_prefixes", "prefixes"),
    ("tool_doc_allowed_apps", "doc_allowed_apps"),
)
_IGNORED_ENTRY_FIELDS = {"observability_events"}


def build_support_slot_id(step: Any, ordinal: int) -> str:
    """Build a deterministic support slot id from declaration order and adapter."""

    token = get_step_adapter_id(step) or get_step_type(step) or "support"
    normalized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(token))
    return f"support:{max(0, int(ordinal)):02d}:{normalized}"


def record_support_output(
    sample: MutableMapping[str, Any],
    *,
    slot_id: str,
    adapter_id: Optional[str],
    output: Mapping[str, Any],
    policy: Optional[Mapping[str, Any]] = None,
) -> None:
    """Record a support output using the normalized artifact envelope."""

    projection_mode = _resolve_projection_mode(policy)
    artifacts = _ensure_support_artifacts(sample, projection_mode)
    entries = list(artifacts.get("entries") or [])
    entry = _build_entry(
        slot_id=slot_id,
        adapter_id=adapter_id,
        output=output,
    )
    replaced = False
    for index, existing in enumerate(entries):
        if isinstance(existing, dict) and existing.get("slot_id") == slot_id:
            entries[index] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)

    artifacts["entries"] = entries
    _refresh_support_artifacts(sample, artifacts, projection_mode=projection_mode)

    if projection_mode == "legacy_full":
        support_outputs = sample.setdefault(SUPPORT_OUTPUTS_KEY, [])
        if isinstance(support_outputs, list):
            support_outputs.append(dict(output))
        return
    if projection_mode == "none":
        sample.pop(SUPPORT_OUTPUTS_KEY, None)
        return
    sample[SUPPORT_OUTPUTS_KEY] = project_support_outputs(sample)


def project_support_outputs(sample: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Project compact compatibility outputs from the normalized artifacts."""

    artifacts = sample.get(SUPPORT_ARTIFACTS_KEY)
    if not isinstance(artifacts, Mapping):
        outputs = sample.get(SUPPORT_OUTPUTS_KEY)
        if isinstance(outputs, list):
            return [dict(item) for item in outputs if isinstance(item, dict)]
        return []

    outputs: List[Dict[str, Any]] = []
    for entry in artifacts.get("entries") or []:
        if not isinstance(entry, Mapping):
            continue
        projected: Dict[str, Any] = {}
        slot_id = entry.get("slot_id")
        adapter_id = entry.get("adapter_id")
        if slot_id:
            projected["slot_id"] = str(slot_id)
        if adapter_id:
            projected["adapter_id"] = str(adapter_id)
        fields = entry.get("fields")
        if isinstance(fields, Mapping):
            projected.update(fields)
        outputs.append(projected)
    return outputs


def iter_support_outputs(sample: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return support outputs using artifacts when available."""

    artifacts = sample.get(SUPPORT_ARTIFACTS_KEY)
    if isinstance(artifacts, Mapping):
        return project_support_outputs(sample)
    outputs = sample.get(SUPPORT_OUTPUTS_KEY)
    if isinstance(outputs, list):
        return [dict(item) for item in outputs if isinstance(item, dict)]
    return []


def resolve_support_field(sample: Mapping[str, Any], field_name: str) -> Any:
    """Resolve a latest support field from artifacts or legacy outputs."""

    artifacts = sample.get(SUPPORT_ARTIFACTS_KEY)
    if isinstance(artifacts, Mapping):
        latest_fields = artifacts.get("latest_fields")
        if isinstance(latest_fields, Mapping) and field_name in latest_fields:
            return copy.deepcopy(latest_fields[field_name])

    latest = None
    for output in iter_support_outputs(sample):
        if field_name in output:
            latest = output.get(field_name)
    return copy.deepcopy(latest)


def resolve_support_tools(sample: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return merged tools from artifacts or legacy outputs."""

    artifacts = sample.get(SUPPORT_ARTIFACTS_KEY)
    if isinstance(artifacts, Mapping):
        merged_tools = artifacts.get("merged_tools")
        if isinstance(merged_tools, list):
            return copy.deepcopy([tool for tool in merged_tools if isinstance(tool, dict)])

    merged: List[Dict[str, Any]] = []
    merged = _merge_tools(merged, sample.get("tools") or [])
    for output in iter_support_outputs(sample):
        merged = _merge_tools(merged, output.get("tools_schema") or output.get("tools") or [])
    return merged


def resolve_support_filters(sample: Mapping[str, Any]) -> Dict[str, Any]:
    """Return merged dynamic filters from artifacts or legacy outputs."""

    artifacts = sample.get(SUPPORT_ARTIFACTS_KEY)
    if isinstance(artifacts, Mapping):
        dynamic_filters = artifacts.get("dynamic_filters")
        if isinstance(dynamic_filters, Mapping):
            resolved = dict(dynamic_filters)
            resolved["allowlist"] = list(resolved.get("allowlist") or [])
            resolved["prefixes"] = list(resolved.get("prefixes") or [])
            resolved["doc_allowed_apps"] = list(resolved.get("doc_allowed_apps") or [])
            return resolved

    allowlist: List[str] = []
    prefixes: List[str] = []
    doc_allowed_apps: List[str] = []
    max_tools: Optional[int] = None
    for output in iter_support_outputs(sample):
        _extend_unique(allowlist, _normalize_string_list(output.get("tool_allowlist")))
        _extend_unique(prefixes, _normalize_string_list(output.get("tool_prefixes")))
        _extend_unique(doc_allowed_apps, _normalize_string_list(output.get("tool_doc_allowed_apps")))
        candidate = _coerce_int(output.get("tool_max_tools"))
        if candidate is not None:
            max_tools = candidate if max_tools is None else min(max_tools, candidate)
    return {
        "allowlist": allowlist,
        "prefixes": prefixes,
        "doc_allowed_apps": doc_allowed_apps,
        "max_tools": max_tools,
    }


def _ensure_support_artifacts(
    sample: MutableMapping[str, Any],
    projection_mode: str,
) -> Dict[str, Any]:
    artifacts = sample.get(SUPPORT_ARTIFACTS_KEY)
    if isinstance(artifacts, dict):
        artifacts.setdefault("version", SUPPORT_ARTIFACTS_VERSION)
        artifacts.setdefault("entries", [])
        artifacts.setdefault("latest_fields", {})
        artifacts.setdefault("merged_tools", [])
        artifacts.setdefault("dynamic_filters", {})
        artifacts.setdefault("stats", {})
        artifacts["stats"].setdefault("projection_mode", projection_mode)
        return artifacts
    artifacts = {
        "version": SUPPORT_ARTIFACTS_VERSION,
        "entries": [],
        "latest_fields": {},
        "merged_tools": [],
        "dynamic_filters": {},
        "stats": {"projection_mode": projection_mode},
    }
    sample[SUPPORT_ARTIFACTS_KEY] = artifacts
    return artifacts


def _build_entry(
    *,
    slot_id: str,
    adapter_id: Optional[str],
    output: Mapping[str, Any],
) -> Dict[str, Any]:
    fields = {
        key: copy.deepcopy(value)
        for key, value in dict(output).items()
        if key not in _IGNORED_ENTRY_FIELDS and value is not None
    }
    entry: Dict[str, Any] = {
        "slot_id": slot_id,
        "fields": fields,
        "stats": {
            "tool_count": _count_tools(fields),
        },
    }
    if adapter_id:
        entry["adapter_id"] = adapter_id
    return entry


def _refresh_support_artifacts(
    sample: Mapping[str, Any],
    artifacts: MutableMapping[str, Any],
    *,
    projection_mode: str,
) -> None:
    entries = [entry for entry in artifacts.get("entries") or [] if isinstance(entry, dict)]
    latest_fields: Dict[str, Any] = {}
    allowlist: List[str] = []
    prefixes: List[str] = []
    doc_allowed_apps: List[str] = []
    max_tools: Optional[int] = None
    merged_tools: List[Dict[str, Any]] = []
    merged_tools = _merge_tools(merged_tools, sample.get("tools") or [])
    for entry in entries:
        fields = entry.get("fields")
        if not isinstance(fields, Mapping):
            continue
        for field_name in _LATEST_FIELD_NAMES:
            if field_name in fields:
                latest_fields[field_name] = copy.deepcopy(fields[field_name])
        _extend_unique(allowlist, _normalize_string_list(fields.get("tool_allowlist")))
        _extend_unique(prefixes, _normalize_string_list(fields.get("tool_prefixes")))
        _extend_unique(doc_allowed_apps, _normalize_string_list(fields.get("tool_doc_allowed_apps")))
        candidate = _coerce_int(fields.get("tool_max_tools"))
        if candidate is not None:
            max_tools = candidate if max_tools is None else min(max_tools, candidate)
        merged_tools = _merge_tools(merged_tools, fields.get("tools_schema") or fields.get("tools") or [])

    artifacts["version"] = SUPPORT_ARTIFACTS_VERSION
    artifacts["entries"] = entries
    artifacts["latest_fields"] = latest_fields
    artifacts["merged_tools"] = merged_tools
    artifacts["dynamic_filters"] = {
        "allowlist": allowlist,
        "prefixes": prefixes,
        "doc_allowed_apps": doc_allowed_apps,
        "max_tools": max_tools,
    }
    artifacts["stats"] = {
        "entry_count": len(entries),
        "projection_mode": projection_mode,
        "tool_count": len(merged_tools),
        "blob_spills": 0,
    }


def _resolve_projection_mode(policy: Optional[Mapping[str, Any]]) -> str:
    if isinstance(policy, Mapping):
        candidate = policy.get("projection_mode")
        if isinstance(candidate, str) and candidate in VALID_PROJECTION_MODES:
            return candidate
    return DEFAULT_PROJECTION_MODE


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Iterable):
        normalized: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized
    return []


def _merge_tools(existing: List[Dict[str, Any]], tools: Any) -> List[Dict[str, Any]]:
    ordered: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    anonymous: List[Dict[str, Any]] = []
    for tool in existing:
        normalized = _normalize_tool_entry(tool)
        if not normalized:
            continue
        name = _extract_tool_name(normalized)
        if name is None:
            anonymous.append(normalized)
            continue
        ordered[name] = normalized

    if isinstance(tools, Mapping):
        tools = [tools]
    for tool in tools or []:
        normalized = _normalize_tool_entry(tool)
        if not normalized:
            continue
        name = _extract_tool_name(normalized)
        if name is None:
            anonymous.append(normalized)
            continue
        if name in ordered:
            ordered.pop(name)
        ordered[name] = normalized
    return [*ordered.values(), *anonymous]


def _normalize_tool_entry(tool: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(tool, Mapping):
        return None
    x_gage = tool.get("x-gage")
    if tool.get("type") == "function" and "function" in tool:
        normalized = dict(tool)
        function = normalized.get("function")
        if isinstance(function, Mapping):
            function_dict = dict(function)
            function_dict["description"] = function_dict.get("description") or ""
            if "parameters" in function_dict:
                function_dict["parameters"] = _strip_nulls(function_dict.get("parameters") or {})
            normalized["function"] = function_dict
        if x_gage is not None and "x-gage" not in normalized:
            normalized["x-gage"] = x_gage
        return normalized
    if "name" in tool and ("parameters" in tool or "inputSchema" in tool):
        normalized = {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description") or "",
                "parameters": _strip_nulls(tool.get("parameters") or tool.get("inputSchema") or {}),
            },
        }
        if x_gage is not None:
            normalized["x-gage"] = x_gage
        return normalized
    normalized = dict(tool)
    if x_gage is not None and "x-gage" not in normalized:
        normalized["x-gage"] = x_gage
    return normalized


def _extract_tool_name(tool: Mapping[str, Any]) -> Optional[str]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), Mapping):
        name = tool["function"].get("name")
        return str(name) if name else None
    name = tool.get("name")
    return str(name) if name else None


def _count_tools(fields: Mapping[str, Any]) -> int:
    tools = fields.get("tools_schema") or fields.get("tools")
    if isinstance(tools, list):
        return len([item for item in tools if isinstance(item, Mapping)])
    if isinstance(tools, Mapping):
        return 1
    return 0


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extend_unique(target: List[str], values: Sequence[str]) -> None:
    existing = set(target)
    for value in values:
        if value in existing:
            continue
        target.append(value)
        existing.add(value)


def _strip_nulls(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            cleaned[str(key)] = _strip_nulls(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_nulls(item) for item in value if item is not None]
    return value
