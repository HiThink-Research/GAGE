"""Chat message normalization helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def stringify_message_content(
    content: Any,
    *,
    image_placeholder: Optional[str] = "<image>",
    coerce_non_text_fragments: bool = True,
) -> str:
    """
    Convert chat message content into a single text string.

    - Flattens list-based fragments (e.g., [{"type": "text", "text": "hi"}, {"type": "image"}]).
    - Replaces image/image_url fragments with the provided placeholder (skips when set to None/"").
    - Coerces non-dict fragments to strings when requested.
    """

    if isinstance(content, list):
        parts: List[str] = []
        for fragment in content:
            if isinstance(fragment, dict):
                frag_type = fragment.get("type")
                if frag_type == "text":
                    text = fragment.get("text")
                    if text is not None:
                        parts.append(str(text))
                elif frag_type in {"image", "image_url"} and image_placeholder:
                    parts.append(image_placeholder)
            elif coerce_non_text_fragments and fragment is not None:
                parts.append(str(fragment))
        return " ".join(part for part in parts if part).strip()

    if content is None:
        return ""
    return str(content)


def normalize_messages_for_template(
    messages: List[Dict[str, Any]],
    *,
    image_placeholder: Optional[str] = "<image>",
    coerce_non_text_fragments: bool = True,
) -> List[Dict[str, Any]]:
    """Return a copy of messages where list-based content is stringified for chat templates."""

    normalized: List[Dict[str, Any]] = []
    for message in messages or []:
        item = dict(message)
        item["content"] = stringify_message_content(
            item.get("content"),
            image_placeholder=image_placeholder,
            coerce_non_text_fragments=coerce_non_text_fragments,
        )
        normalized.append(item)
    return normalized
