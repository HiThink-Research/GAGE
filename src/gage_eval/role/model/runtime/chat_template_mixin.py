"""Shared chat template utilities for backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional


TemplateFn = Callable[..., Any]
FallbackFn = Callable[[Optional[list]], str]


@dataclass
class ChatTemplatePolicy:
    mode: Literal["auto", "never", "force"] = "auto"
    source: Literal["model", "fallback"] = "model"
    rendered_by: Optional[Literal["preprocess", "backend", "none"]] = None


@dataclass
class BackendCapabilities:
    supports_mm: bool = False
    has_processor_chat_template: bool = False


class ChatTemplateMixin:
    """Provide shared helpers for chat template decision and rendering."""

    @staticmethod
    def detect_multimodal(payload: Dict[str, Any]) -> bool:
        """Detect multimodal content from messages or multi_modal_data."""

        messages = payload.get("messages") or []
        mm = payload.get("multi_modal_data") or {}
        if mm:
            return True
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for fragment in content:
                    if isinstance(fragment, dict) and fragment.get("type") in {"image", "image_url", "audio", "audio_url"}:
                        return True
        return False

    @staticmethod
    def should_render(payload: Dict[str, Any], policy: ChatTemplatePolicy) -> bool:
        """Guard against double rendering or explicit never mode."""

        mode = payload.get("chat_template_mode")
        rendered_by = payload.get("rendered_by")
        template_source = payload.get("template_source")

        if policy.mode == "never":
            return False
        if mode == "plain":
            return False

        # NOTE: If preprocessing only applied a fallback concatenation, allow the
        # backend to re-render with the model template.
        if mode == "preprocess" or rendered_by == "preprocess":
            if template_source == "fallback":
                return True
            return False

        return True

    @staticmethod
    def select_template(kind: Literal["text", "vlm"], policy: ChatTemplatePolicy, caps: BackendCapabilities) -> str:
        """Decide template source: model or fallback."""

        if policy.source == "fallback":
            return "fallback"
        if kind == "vlm":
            return "model" if caps.has_processor_chat_template else "fallback"
        return "model"

    @staticmethod
    def render(
        messages: Optional[list],
        template_fn: Optional[TemplateFn],
        fallback_fn: FallbackFn,
        add_generation_prompt: bool = True,
        chat_template: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Render messages using template_fn with fallback."""

        messages = messages or []
        if template_fn is not None:
            try:
                rendered = template_fn(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    chat_template=chat_template,
                    **kwargs,
                )
                if rendered:
                    return str(rendered if not isinstance(rendered, list) else rendered[0])
            except Exception:  # pragma: no cover - defensive fallback
                pass
        return fallback_fn(messages)

    @staticmethod
    def get_cache_suffix(kind: Literal["text", "vlm"], policy: ChatTemplatePolicy, caps: BackendCapabilities) -> str:
        """Generate a deterministic cache suffix based on policy and capabilities."""

        if policy.mode == "never":
            return "-plain"
        if kind == "vlm":
            return "-processor" if caps.has_processor_chat_template and policy.source == "model" else "-fallback"
        return "-chat_template" if policy.source == "model" else "-fallback"
