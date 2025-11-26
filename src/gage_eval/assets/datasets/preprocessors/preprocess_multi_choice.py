"""Preprocessor that converts raw multiple-choice samples into chat prompts."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

_CHOICE_ALPHABET: Tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_TOKEN_SPLIT = re.compile(r"[^A-Z]+")


def convert_sample_to_inputs(
    sample: Dict[str, Any],
    *,
    question_field: str = "question",
    options_field: str | None = None,
    choices_field: str | None = None,
    answer_field: str = "answer",
    answer_index_base: int = 0,
    system_prompt: str | None = None,
    instruction: str | None = "请仅输出正确选项对应的大写字母，例如 'A'。",
) -> List[Dict[str, Any]]:
    """Normalize multiple-choice samples and attach option metadata."""

    question = _extract_nested(sample, question_field)
    if question is None:
        raise ValueError(f"Multiple-choice sample missing question field '{question_field}'")

    option_source = choices_field or options_field or "choices"
    raw_options = _extract_nested(sample, option_source)
    options = _normalize_options(raw_options)
    if len(options) < 2:
        raise ValueError("Multiple-choice sample must provide at least two options")

    option_pairs = [(_CHOICE_ALPHABET[idx], text) for idx, text in enumerate(options)]
    if len(option_pairs) > len(_CHOICE_ALPHABET):
        raise ValueError("Multiple-choice preprocessor supports up to 26 options")

    answer_value = _extract_nested(sample, answer_field)
    correct_choice = _resolve_correct_choice(answer_value, option_pairs, answer_index_base)
    if correct_choice is None:
        raise ValueError("Unable to resolve correct choice from answer field")

    user_prompt = _compose_prompt(question, option_pairs, instruction)
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt.strip(),
                    }
                ],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                }
            ],
        }
    )

    sample["messages"] = messages
    sample["prompt"] = user_prompt
    sample["choices"] = [
        {
            "index": idx,
            "label": label,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": option_text,
                    }
                ],
            },
        }
        for idx, (label, option_text) in enumerate(option_pairs)
    ]
    metadata = dict(sample.get("metadata") or {})
    metadata.update(
        {
            "option_map": {label: option_text for label, option_text in option_pairs},
            "correct_choice": correct_choice,
            "question_field": question_field,
            "choices_field": option_source,
            "answer_field": answer_field,
        }
    )
    sample["metadata"] = metadata
    return messages


def _extract_nested(sample: Dict[str, Any], field: str | None) -> Any:
    if not field:
        return None

    def _traverse(obj: Any, path: str) -> Any:
        current: Any = obj
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    index = int(part)
                except ValueError:
                    return None
                if index < 0 or index >= len(current):
                    return None
                current = current[index]
            else:
                return None
            if current is None:
                return None
        return current

    for candidate in field.split("|"):
        trimmed = candidate.strip()
        if not trimmed:
            continue
        result = _traverse(sample, trimmed)
        if result is not None:
            return result
    return None


def _normalize_options(options: Any) -> List[str]:
    if options is None:
        return []
    if isinstance(options, dict):
        sorted_items = sorted(options.items())
        return [str(value).strip() for _, value in sorted_items]
    if isinstance(options, str):
        return [opt for opt in options.splitlines() if opt.strip()]
    if isinstance(options, Sequence):
        normalized: List[str] = []
        for entry in options:
            if isinstance(entry, str):
                normalized.append(entry.strip())
            elif isinstance(entry, dict):
                text = entry.get("text") or entry.get("label") or entry.get("choice")
                if text is not None:
                    normalized.append(str(text).strip())
            else:
                normalized.append(str(entry).strip())
        return normalized
    return [str(options).strip()]


def _resolve_correct_choice(
    answer_value: Any,
    option_pairs: Sequence[Tuple[str, str]],
    answer_index_base: int,
) -> str | None:
    letters = [label for label, _ in option_pairs]
    if answer_value is None:
        return None
    if isinstance(answer_value, bool):
        return None
    if isinstance(answer_value, (int, float)):
        index = int(answer_value) - answer_index_base
        return letters[index] if 0 <= index < len(letters) else None
    answer_str = str(answer_value).strip()
    if not answer_str:
        return None
    if answer_str.isdigit():
        index = int(answer_str) - answer_index_base
        return letters[index] if 0 <= index < len(letters) else None
    normalized_letter = _normalize_choice_label(answer_str, letters)
    if normalized_letter:
        return normalized_letter
    normalized_text = _normalize_text(answer_str)
    for label, option_text in option_pairs:
        if normalized_text == _normalize_text(option_text):
            return label
    return None


def _compose_prompt(question: Any, option_pairs: Sequence[Tuple[str, str]], instruction: str | None) -> str:
    question_block = str(question).strip()
    option_lines = "\n".join(f"{label}. {text}" for label, text in option_pairs)
    prompt_parts = [question_block, "", option_lines]
    if instruction:
        prompt_parts.extend(["", instruction.strip()])
    return "\n".join(part for part in prompt_parts if part is not None).strip()


def _normalize_choice_label(value: str, allowed_letters: Sequence[str]) -> str | None:
    candidate = value.upper()
    cleaned = candidate.strip(" .:：()[]{}<>\"'")
    if cleaned in allowed_letters:
        return cleaned
    for prefix in ("OPTION", "CHOICE", "ANSWER", "答案", "选项"):
        if cleaned.startswith(prefix):
            stripped = cleaned[len(prefix) :].strip(" .:：()[]{}<>\"'")
            if stripped in allowed_letters:
                return stripped
    for token in _TOKEN_SPLIT.split(candidate):
        if token in allowed_letters:
            return token
    return None


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


__all__ = ["convert_sample_to_inputs"]
