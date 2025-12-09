"""Field mapping helpers to turn heterogeneous records into Sample-friendly structures."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import re

_CHOICE_ALPHABET: Tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def extract_field(sample: Dict[str, Any], field: Optional[str], default: Any = None) -> Any:
    """Extract nested field using dot path, support '|' fallback."""

    if not field:
        return default

    def _traverse(obj: Any, path: str) -> Any:
        current: Any = obj
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, (list, tuple)):
                try:
                    idx = int(part)
                except (TypeError, ValueError):
                    return default
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return default
            else:
                return default
            if current is None:
                return default
        return current

    for candidate in field.split("|"):
        candidate = candidate.strip()
        if not candidate:
            continue
        result = _traverse(sample, candidate)
        if result is not default and result is not None:
            return result
    return default


def normalize_options(options: Any) -> List[str]:
    """Normalize options into a list of strings."""

    if options is None:
        return []
    if isinstance(options, dict):
        sorted_items = sorted(options.items())
        return [str(value).strip() for _, value in sorted_items if str(value).strip()]
    if isinstance(options, str):
        return [opt.strip() for opt in options.splitlines() if opt.strip()]
    if isinstance(options, Sequence):
        normalized: List[str] = []
        for entry in options:
            if isinstance(entry, str):
                text = entry.strip()
                if text:
                    normalized.append(text)
            elif isinstance(entry, dict):
                text = entry.get("text") or entry.get("label") or entry.get("choice") or entry.get("content")
                if text is not None and str(text).strip():
                    normalized.append(str(text).strip())
            else:
                text = str(entry).strip()
                if text:
                    normalized.append(text)
        return normalized
    text = str(options).strip()
    return [text] if text else []


def resolve_correct_choice(answer_value: Any, option_pairs: Sequence[Tuple[str, str]], answer_index_base: int = 0) -> Optional[str]:
    """Resolve correct choice letter from answer value and option list."""

    letters = [label for label, _ in option_pairs]
    if answer_value is None:
        return None
    if isinstance(answer_value, bool):
        return None
    if isinstance(answer_value, (int, float)):
        idx = int(answer_value) - answer_index_base
        return letters[idx] if 0 <= idx < len(letters) else None

    answer_str = str(answer_value).strip()
    if not answer_str:
        return None
    if answer_str.isdigit():
        idx = int(answer_str) - answer_index_base
        return letters[idx] if 0 <= idx < len(letters) else None

    normalized_letter = _normalize_choice_label(answer_str, letters)
    if normalized_letter:
        return normalized_letter

    normalized_text = _normalize_text(answer_str)
    for label, option_text in option_pairs:
        if normalized_text == _normalize_text(option_text):
            return label
    return None


def map_question_option_answer(
    sample: Dict[str, Any],
    *,
    question_field: str = "question",
    option_field: str = "choices",
    answer_field: str = "answer",
    answer_index_base: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Map question/options/answer to messages, choices, metadata."""

    question = extract_field(sample, question_field)
    options = normalize_options(extract_field(sample, option_field))
    if question is None or not options:
        return [], [], {}

    option_pairs = [(_CHOICE_ALPHABET[idx], text) for idx, text in enumerate(options)]
    correct_choice = resolve_correct_choice(extract_field(sample, answer_field), option_pairs, answer_index_base)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": str(question),
                }
            ],
        }
    ]
    choices = [
        {
            "index": idx,
            "label": label,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": option_text}],
            },
        }
        for idx, (label, option_text) in enumerate(option_pairs)
    ]
    metadata = {
        "option_map": {label: text for label, text in option_pairs},
        "correct_choice": correct_choice,
        "question_field": question_field,
        "choices_field": option_field,
        "answer_field": answer_field,
    }
    return messages, choices, metadata


def _normalize_choice_label(value: str, allowed_letters: Sequence[str]) -> Optional[str]:
    candidate = value.upper()
    cleaned = candidate.strip(" .:：()[]{}<>\"'")
    if cleaned in allowed_letters:
        return cleaned
    for prefix in ("OPTION", "CHOICE", "ANSWER", "答案", "选项"):
        if cleaned.startswith(prefix):
            stripped = cleaned[len(prefix):].strip(" .:：()[]{}<>\"'")
            if stripped in allowed_letters:
                return stripped
    for token in _token_split(candidate):
        if token in allowed_letters:
            return token
    return None


def _token_split(text: str) -> List[str]:
    out: List[str] = []
    current = ""
    for ch in text:
        if ch.isalpha():
            current += ch
        else:
            if current:
                out.append(current)
                current = ""
    if current:
        out.append(current)
    return out


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _expand_env(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} patterns with env fallback."""

    if value is None:
        return value
    pattern = re.compile(r"\$\{([^}:]+):-(.+)\}")
    match = pattern.search(value)
    if match:
        var, default = match.groups()
        if os.environ.get(var):
            return os.path.expandvars(pattern.sub(f"${{{var}}}", value))
        return pattern.sub(default, value)
    return os.path.expandvars(value)
