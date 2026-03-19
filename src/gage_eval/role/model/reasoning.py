"""Utilities for extracting and processing reasoning/thinking content from model responses."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


def extract_reasoning_content(message: Any) -> Optional[str]:
    """Extracts reasoning_content from an OpenAI-style message object or dict.

    Supports both attribute-based access (SDK response objects) and
    dict-based access (raw JSON responses).

    Args:
        message: A message object or dict from a chat completion response.

    Returns:
        The reasoning content string, or None if not present.
    """
    if message is None:
        return None

    # Attribute-based access (e.g. OpenAI SDK ChatCompletionMessage)
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is not None:
        return str(reasoning)

    # Dict-based access
    if isinstance(message, dict):
        reasoning = message.get("reasoning_content")
        if reasoning is not None:
            return str(reasoning)

    return None


_THINK_TAG_PATTERN = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL,
)


def strip_thinking_tags(text: str) -> Tuple[str, Optional[str]]:
    """Strips ``<think>...</think>`` tags from text.

    Some models (e.g. Qwen3 in thinking mode) wrap their chain-of-thought
    reasoning inside ``<think>...</think>`` tags.  This helper separates the
    visible answer from the thinking trace.

    Args:
        text: The raw model output text.

    Returns:
        A tuple of ``(clean_text, thinking_content)``.  ``thinking_content``
        is ``None`` when no ``<think>`` tags are found.
    """
    if not text:
        return text, None

    matches = _THINK_TAG_PATTERN.findall(text)
    if not matches:
        return text, None

    thinking_content = "\n".join(m.strip() for m in matches)
    clean_text = _THINK_TAG_PATTERN.sub("", text).strip()
    return clean_text, thinking_content


def resolve_thinking_kwargs(
    thinking_mode: Optional[str],
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merges a high-level ``thinking_mode`` into ``chat_template_kwargs``.

    The ``thinking_mode`` is a user-friendly enum (``"enabled"`` /
    ``"disabled"``).  This function translates it into the low-level
    ``enable_thinking`` boolean used by tokenizer chat templates (e.g.
    Qwen3).  Explicit values in *extra_kwargs* take precedence.

    Args:
        thinking_mode: ``"enabled"``, ``"disabled"``, or ``None``.
        extra_kwargs: Additional kwargs that may already contain
            ``enable_thinking`` or other template parameters.

    Returns:
        A merged dict suitable for passing as ``chat_template_kwargs``.
    """
    merged: Dict[str, Any] = {}

    if thinking_mode is not None:
        mode_lower = thinking_mode.strip().lower()
        if mode_lower in ("disabled", "off", "false", "no", "0"):
            merged["enable_thinking"] = False
        elif mode_lower in ("enabled", "on", "true", "yes", "1"):
            merged["enable_thinking"] = True

    # Explicit extra_kwargs override the derived value
    if extra_kwargs:
        merged.update(extra_kwargs)

    return merged
