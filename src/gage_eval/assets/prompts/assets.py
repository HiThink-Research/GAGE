"""Prompt asset helpers that bind configs to renderer instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from gage_eval.assets.prompts.renderers import (
    JinjaDelimitedChatPromptRenderer,
    JinjaChatPromptRenderer,
    JinjaPromptRenderer,
    PassthroughPromptRenderer,
    PromptRenderer,
)


def build_prompt_renderer(renderer_type: str, *, template: Optional[str], params: Dict[str, Any]) -> PromptRenderer:
    kind = (renderer_type or "").lower()
    if kind in {"jinja", "jinja2"}:
        return JinjaPromptRenderer(template or "", variables=params)
    if kind in {"jinja_chat", "chat_jinja", "chat"}:
        role = params.get("role", "system")
        include_existing = params.get("include_existing", True)
        return JinjaChatPromptRenderer(
            template or "",
            variables=params,
            role=role,
            include_existing=bool(include_existing),
        )
    if kind in {"jinja_delimited_chat", "delimited_jinja_chat", "delimited_chat"}:
        mode = params.get("mode", "full")
        include_system = params.get("include_system", True)
        message_delimiter = params.get("message_delimiter", "---")
        section_delimiter = params.get("section_delimiter", "===")
        return JinjaDelimitedChatPromptRenderer(
            template or "",
            variables=params,
            mode=mode,
            include_system=bool(include_system),
            message_delimiter=str(message_delimiter or "---"),
            section_delimiter=str(section_delimiter or "==="),
        )
    if kind in {"passthrough", "raw"}:
        fields = params.get("fields")
        default = params.get("default", "")
        return PassthroughPromptRenderer(fields=fields, default=default)
    raise KeyError(f"Unknown prompt renderer type '{renderer_type}'")


@dataclass
class PromptTemplateAsset:
    """Represents a reusable prompt template defined in config."""

    prompt_id: str
    renderer_type: str
    template: Optional[str] = None
    default_args: Dict[str, Any] = field(default_factory=dict)

    def instantiate(self, overrides: Optional[Dict[str, Any]] = None) -> PromptRenderer:
        params = dict(self.default_args)
        if overrides:
            params.update(overrides)
        return build_prompt_renderer(self.renderer_type, template=self.template, params=params)
