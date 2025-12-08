"""Answer parsing and enrichment helpers."""

from __future__ import annotations

import ast
import re
from typing import Any, List, Dict


def parse_list_from_string(raw: Any) -> List[str]:
    """Safely parse list-like answers from mixed inputs."""

    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass
        return [text]
    return [str(raw).strip()]


_CHOICE_PATTERN = re.compile(r"^\s*([A-Z])[\.\)]\s*(.+)$")


def enrich_answer_with_options(question: Any, answers: List[str]) -> List[str]:
    """Expand choice labels (A/B/...) into full option text if present in question."""

    if not answers:
        return answers
    choice_map = _extract_choice_map(question)
    if not choice_map:
        return answers
    enriched: List[str] = []
    seen = set()
    for answer in answers:
        normalized = answer.strip()
        if normalized and normalized not in seen:
            enriched.append(normalized)
            seen.add(normalized)
        label = _normalize_choice_label(normalized)
        if label and label in choice_map:
            option_text = choice_map[label]
            if option_text not in seen:
                enriched.append(option_text)
                seen.add(option_text)
    return enriched


def _extract_choice_map(question: Any) -> Dict[str, str]:
    if question is None:
        return {}
    mapping: Dict[str, str] = {}
    for line in str(question).splitlines():
        match = _CHOICE_PATTERN.match(line.strip())
        if match:
            label = match.group(1).upper()
            text = match.group(2).strip()
            if text:
                mapping[label] = text
    return mapping


def _normalize_choice_label(value: str) -> str | None:
    token = value.strip().upper()
    cleaned = token.strip(" .:：()[]{}\"'")
    if len(cleaned) == 1 and cleaned.isalpha():
        return cleaned
    return None


__all__ = ["parse_list_from_string", "enrich_answer_with_options"]
