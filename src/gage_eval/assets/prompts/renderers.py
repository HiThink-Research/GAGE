"""Prompt renderer implementations and shared dataclasses."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class PromptContext:
    """Carries all contextual information required to render a prompt."""

    sample: Dict[str, Any]
    payload: Dict[str, Any] = field(default_factory=dict)
    history: Sequence[Dict[str, Any]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> Dict[str, Any]:
        """Return a mapping that templates can consume safely."""

        tool_documentation, tool_documentation_meta = _extract_tool_documentation(self.sample, self.payload)
        return {
            "sample": _safe_copy(self.sample),
            "payload": _safe_copy(self.payload),
            "history": _safe_copy(list(self.history)),
            "extras": _safe_copy(self.extras),
            "tool_documentation": tool_documentation,
            "tool_documentation_meta": _safe_copy(tool_documentation_meta),
        }


def _safe_copy(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return tuple(value)
        return value


def _extract_tool_documentation(sample: Dict[str, Any], payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    for source in (payload, sample):
        if isinstance(source, dict):
            doc = source.get("tool_documentation")
            meta = source.get("tool_documentation_meta") if isinstance(source.get("tool_documentation_meta"), dict) else {}
            if isinstance(doc, str) and doc.strip():
                return doc, dict(meta)
    outputs = []
    if isinstance(payload, dict):
        outputs.extend(payload.get("support_outputs") or [])
    if isinstance(sample, dict):
        outputs.extend(sample.get("support_outputs") or [])
    for output in outputs:
        if not isinstance(output, dict):
            continue
        doc = output.get("tool_documentation")
        meta = output.get("tool_documentation_meta") if isinstance(output.get("tool_documentation_meta"), dict) else {}
        if isinstance(doc, str) and doc.strip():
            return doc, dict(meta)
    return "", {}


@dataclass
class PromptRenderResult:
    """Structured prompt output returned by renderers."""

    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> Dict[str, Any]:
        """Convert the result into a backend-friendly payload fragment."""

        payload: Dict[str, Any] = {}
        if self.prompt is not None:
            payload["prompt"] = self.prompt
        if self.messages is not None:
            payload["messages"] = self.messages
        if self.metadata:
            payload["prompt_meta"] = self.metadata
        return payload


class PromptRenderer:
    """Base renderer contract."""

    def render(self, context: PromptContext) -> PromptRenderResult:  # pragma: no cover - abstract
        raise NotImplementedError


class PassthroughPromptRenderer(PromptRenderer):
    """Simple renderer that reuses common fields from the sample/payload."""

    def __init__(self, fields: Optional[Sequence[str]] = None, default: str = "") -> None:
        self._fields = list(fields or ("prompt", "text", "question"))
        self._default = default

    def render(self, context: PromptContext) -> PromptRenderResult:
        for field in self._fields:
            if field in context.payload:
                value = context.payload[field]
            else:
                value = context.sample.get(field)
            if isinstance(value, str) and value.strip():
                return PromptRenderResult(prompt=value)
        return PromptRenderResult(prompt=self._default)


class JinjaPromptRenderer(PromptRenderer):
    """Render prompts using a Jinja2 template."""

    def __init__(self, template: str, variables: Optional[Dict[str, Any]] = None) -> None:
        try:
            from jinja2 import Environment, StrictUndefined
        except ImportError as exc:  # pragma: no cover - exercised in runtime
            raise RuntimeError(
                "JinjaPromptRenderer requires the 'jinja2' package. "
                "Please install it via 'pip install jinja2'."
            ) from exc

        self._env = Environment(autoescape=False, undefined=StrictUndefined)
        self._template = self._env.from_string(template or "")
        self._defaults = variables or {}

    def render(self, context: PromptContext) -> PromptRenderResult:
        mapping = context.to_mapping()
        mapping.update(self._defaults)
        prompt = self._template.render(mapping)
        return PromptRenderResult(prompt=prompt.strip())


class JinjaChatPromptRenderer(PromptRenderer):
    """Render a system prompt and prepend it to existing chat messages."""

    def __init__(
        self,
        template: str,
        variables: Optional[Dict[str, Any]] = None,
        *,
        role: str = "system",
        include_existing: bool = True,
    ) -> None:
        try:
            from jinja2 import Environment, StrictUndefined
        except ImportError as exc:  # pragma: no cover - exercised in runtime
            raise RuntimeError(
                "JinjaChatPromptRenderer requires the 'jinja2' package. "
                "Please install it via 'pip install jinja2'."
            ) from exc

        self._env = Environment(autoescape=False, undefined=StrictUndefined)
        self._template = self._env.from_string(template or "")
        self._defaults = variables or {}
        self._role = role
        self._include_existing = include_existing

    def render(self, context: PromptContext) -> PromptRenderResult:
        mapping = context.to_mapping()
        mapping.update(self._defaults)
        system_text = self._template.render(mapping).strip()
        messages: List[Dict[str, Any]] = []
        if system_text:
            messages.append({"role": self._role, "content": system_text})
        if self._include_existing:
            existing = context.payload.get("messages") or context.sample.get("messages") or []
            if isinstance(existing, list):
                messages.extend(copy.deepcopy(existing))
        return PromptRenderResult(messages=messages)
