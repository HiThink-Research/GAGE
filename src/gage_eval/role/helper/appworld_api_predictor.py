"""AppWorld API predictor helper implementation."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from gage_eval.registry import registry


_API_PATTERN = re.compile(r"([A-Za-z0-9_]+)(?:\.|__)([A-Za-z0-9_]+)")


@registry.asset(
    "helper_impls",
    "appworld_api_predictor",
    desc="AppWorld API predictor helper implementation",
    tags=("appworld", "helper"),
)
class AppWorldApiPredictor:
    """Build multi-turn prompts and parse API allowlists for AppWorld."""

    def __init__(
        self,
        *,
        max_apis: int = 16,
        tool_name_separator: str = "__",
        always_include_apps: Optional[Iterable[str]] = None,
        always_include_tools: Optional[Iterable[str]] = None,
    ) -> None:
        self._max_apis = max(1, int(max_apis))
        self._tool_name_separator = tool_name_separator or "__"
        self._always_include_apps = _normalize_items(always_include_apps, lower=True)
        self._always_include_tools = _normalize_items(always_include_tools)

    def prepare_request(self, payload: Dict[str, Any], *, adapter: Any) -> Dict[str, Any]:
        """Prepare the API predictor prompt using the template."""

        sample = payload.get("sample", {}) if isinstance(payload, dict) else {}
        instruction = _extract_instruction(sample)
        api_descriptions = _extract_api_descriptions(sample)
        enriched = dict(payload or {})
        enriched["api_descriptions_string"] = api_descriptions
        enriched["instruction"] = instruction
        enriched["max_apis"] = self._max_apis
        return adapter.build_backend_request(enriched)

    def handle_response(
        self,
        payload: Dict[str, Any],
        response: Dict[str, Any],
        *,
        adapter: Any,
    ) -> Dict[str, Any]:
        """Parse predicted APIs and emit tool filters for downstream toolchain."""

        text = _extract_text(response)
        predicted, truncated = _parse_api_names(text, max_apis=self._max_apis)
        allowed = _extract_allowed_api_info(payload.get("sample") or {})
        if allowed.api_names:
            predicted = [name for name in predicted if name in allowed.api_names]
        tool_allowlist = [_to_tool_name(item, self._tool_name_separator) for item in predicted]
        tool_allowlist.extend(
            _to_tool_name(item, self._tool_name_separator) for item in self._always_include_tools
        )
        tool_allowlist = _dedupe_items(tool_allowlist)
        if allowed.tool_names:
            tool_allowlist = [name for name in tool_allowlist if name in allowed.tool_names]
        predicted_apps = [item.split(".", 1)[0] for item in predicted if "." in item]
        tool_prefixes = _dedupe_items(self._always_include_apps)
        if allowed.app_names:
            predicted_apps = [app for app in predicted_apps if app in allowed.app_names]
            tool_prefixes = [app for app in tool_prefixes if app in allowed.app_names]
            doc_allowed_apps = _dedupe_items([*predicted_apps, *tool_prefixes])
        else:
            doc_allowed_apps = _dedupe_items([*predicted_apps, *self._always_include_apps])

        result: Dict[str, Any] = {
            "api_predictor_output": text,
            "predicted_apis": predicted,
        }
        if tool_allowlist:
            result["tool_allowlist"] = tool_allowlist
        if tool_prefixes:
            result["tool_prefixes"] = tool_prefixes
        if doc_allowed_apps:
            result["tool_doc_allowed_apps"] = doc_allowed_apps
        result["observability_events"] = [
            {
                "event": "api_predictor_result",
                "payload": {
                    "predicted_count": len(predicted),
                    "allowlist_count": len(tool_allowlist),
                    "prefix_count": len(tool_prefixes),
                    "filtered_by_allowed": bool(allowed.api_names or allowed.tool_names),
                    "truncated": truncated,
                    "max_apis": self._max_apis,
                },
            }
        ]
        return result


def _extract_text(response: Any) -> str:
    if isinstance(response, dict):
        for key in ("answer", "response", "text", "content", "output"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else None
            if first:
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
    if isinstance(response, str):
        return response.strip()
    return str(response)


def _parse_api_names(text: str, *, max_apis: int) -> Tuple[List[str], bool]:
    if not text:
        return [], False
    items: List[str] = []
    truncated = False
    for line in text.splitlines():
        for chunk in _split_candidates(line):
            match = _API_PATTERN.search(chunk)
            if not match:
                continue
            app = match.group(1).strip().lower()
            api = match.group(2).strip().lower()
            if not app or not api:
                continue
            name = f"{app}.{api}"
            if name in items:
                continue
            items.append(name)
            if len(items) >= max_apis:
                truncated = True
                return items, truncated
    return items, truncated


def _split_candidates(line: str) -> Iterable[str]:
    cleaned = line.strip()
    if not cleaned:
        return []
    cleaned = re.sub(r"^[-*\s\d).:]+", "", cleaned)
    if "," in cleaned:
        return [part.strip() for part in cleaned.split(",") if part.strip()]
    return [cleaned]


def _to_tool_name(name: str, separator: str) -> str:
    if "__" in name:
        return name
    if "." in name:
        app, api = name.split(".", 1)
        return f"{app}{separator}{api}"
    return name


def _extract_instruction(sample: Dict[str, Any]) -> str:
    instruction = sample.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    prompt = sample.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0] if isinstance(messages[0], dict) else None
        if first:
            content = first.get("content")
            if isinstance(content, list) and content:
                text = content[0].get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _extract_api_descriptions(sample: Dict[str, Any]) -> str:
    outputs = sample.get("support_outputs") or []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        context = output.get("api_descriptions_context")
        if isinstance(context, str) and context.strip():
            return context.strip()
    return ""


class _AllowedApiInfo:
    def __init__(
        self,
        *,
        api_names: List[str],
        tool_names: List[str],
        app_names: List[str],
    ) -> None:
        self.api_names = api_names
        self.tool_names = tool_names
        self.app_names = app_names


def _extract_allowed_api_info(sample: Dict[str, Any]) -> _AllowedApiInfo:
    api_names: List[str] = []
    tool_names: List[str] = []
    app_names: List[str] = []
    outputs = sample.get("support_outputs") or []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        allowed_apis = output.get("api_descriptions_allowed_apis")
        if isinstance(allowed_apis, list):
            api_names.extend([str(item).lower() for item in allowed_apis if item])
        allowed_tools = output.get("api_descriptions_allowed_tools")
        if isinstance(allowed_tools, list):
            tool_names.extend([str(item) for item in allowed_tools if item])
    if api_names:
        app_names.extend([item.split(".", 1)[0] for item in api_names if "." in item])
    if tool_names:
        app_names.extend([item.split("__", 1)[0] for item in tool_names if "__" in item])
    return _AllowedApiInfo(
        api_names=_dedupe_items(api_names),
        tool_names=_dedupe_items(tool_names),
        app_names=_dedupe_items(app_names),
    )


def _normalize_items(items: Optional[Iterable[str]], *, lower: bool = False) -> List[str]:
    if not items:
        return []
    normalized: List[str] = []
    for item in items:
        if not item:
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized.append(text.lower() if lower else text)
    return normalized


def _dedupe_items(items: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped




__all__ = ["AppWorldApiPredictor"]
