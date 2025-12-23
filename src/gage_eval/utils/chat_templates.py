"""Fallback chat templates shared by preprocessors and backends."""

from __future__ import annotations

from typing import Literal

DEFAULT_TEXT_CHAT_TEMPLATE = "{{#each messages}}{{role}}: {{content}}\\n{{/each}}assistant:"
DEFAULT_VLM_CHAT_TEMPLATE = "{{#each messages}}{{role}}: {{content}}\\n{{/each}}assistant:"


def get_fallback_template(kind: Literal["text", "vlm"] = "text") -> str:
    """Return framework-level fallback chat template."""

    if kind == "text":
        return DEFAULT_TEXT_CHAT_TEMPLATE
    if kind == "vlm":
        return DEFAULT_VLM_CHAT_TEMPLATE
    # defensive fallback
    return DEFAULT_TEXT_CHAT_TEMPLATE
