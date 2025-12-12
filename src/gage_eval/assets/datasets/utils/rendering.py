"""Rendering helpers for chat messages."""

from __future__ import annotations

from typing import Any, List, Tuple


def contains_multimodal(messages: list) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for fragment in content:
                if isinstance(fragment, dict) and fragment.get("type") in {"image", "image_url", "audio_url", "video_url"}:
                    return True
    return False


def render_messages_with_fallback(messages: list, tokenizer) -> Tuple[str, str]:
    """Render messages via tokenizer.apply_chat_template if possible, otherwise simple concat."""

    def _normalize_messages(msgs: list) -> list:
        normalized = []
        for m in msgs:
            item = dict(m)
            content = item.get("content")
            if isinstance(content, list):
                text_parts = []
                for frag in content:
                    if isinstance(frag, dict) and frag.get("type") == "text":
                        text_parts.append(str(frag.get("text", "")))
                item["content"] = " ".join(text_parts)
            elif content is None:
                item["content"] = ""
            normalized.append(item)
        return normalized

    def _simple_render(msgs: list) -> str:
        segs: List[str] = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content")
            if isinstance(content, list):
                text_parts = []
                for frag in content:
                    if isinstance(frag, dict) and frag.get("type") == "text":
                        text_parts.append(str(frag.get("text", "")))
                text = " ".join(text_parts)
            else:
                text = str(content) if content is not None else ""
            segs.append(f"{role}: {text}".strip())
        segs.append("assistant:")
        return "\n".join(segs)

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        norm_messages = _normalize_messages(messages)
        try:
            rendered = tokenizer.apply_chat_template(
                norm_messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(rendered, list):
                rendered = rendered[0]
            if rendered:
                return str(rendered), "model"
        except Exception:
            pass
    return _simple_render(messages), "fallback"


__all__ = ["contains_multimodal", "render_messages_with_fallback"]
