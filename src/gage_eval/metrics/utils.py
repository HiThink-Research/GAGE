"""Shared helpers for metric field extraction and text utilities."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional

from loguru import logger

from gage_eval.evaluation.sample_envelope import resolve_judge_output, resolve_model_output

_MISSING = object()


def walk_path(source: Any, descriptor: Optional[str], default: Any = None) -> Any:
    """Traverse nested mappings/lists using dot-separated path segments."""

    if source is None or descriptor in (None, ""):
        return source if descriptor in (None, "") else default

    current = source
    for segment in descriptor.split("."):
        if isinstance(current, Mapping) and segment in current:
            current = current[segment]
            continue
        if isinstance(current, (list, tuple)):
            try:
                idx = int(segment)
            except (TypeError, ValueError):
                return default
            if 0 <= idx < len(current):
                current = current[idx]
                continue
            return _MISSING
        return _MISSING
    return current


def extract_field(context: Any, descriptor: Optional[str], default: Any = None, policy: Optional[str] = None) -> Any:
    """Extract a field from MetricContext or mapping, supporting list indices."""

    if descriptor is None or descriptor == "":
        return default

    sample = None
    roots = {}
    if hasattr(context, "sample"):
        sample = getattr(context, "sample", None)
        roots = {
            "sample": sample,
            "model_output": getattr(context, "model_output", None),
            "judge_output": getattr(context, "judge_output", None),
        }
    elif isinstance(context, Mapping):
        sample = context
        roots = {"sample": context}
    else:
        sample = context

    parts = descriptor.split(".")
    root_key = parts[0]
    tail = ".".join(parts[1:]) if len(parts) > 1 else None

    if root_key == "model_output":
        base = resolve_model_output(sample, roots.get("model_output"))
    elif root_key == "judge_output":
        base = resolve_judge_output(sample, roots.get("judge_output"))
    elif root_key in roots and roots[root_key] is not None:
        base = roots[root_key]
    else:
        base = sample
        tail = descriptor
    missing_policy = policy
    if missing_policy is None and hasattr(context, "args"):
        args = getattr(context, "args", {}) or {}
        if isinstance(args, Mapping):
            missing_policy = args.get("on_missing_field")
    missing_policy = (missing_policy or "ignore").lower()

    result = walk_path(base, tail, default=_MISSING)
    if result is not _MISSING:
        return result

    if missing_policy == "error":
        raise KeyError(f"Field '{descriptor}' missing during metric extraction")
    if missing_policy == "warn":
        logger.warning("Metric extract_field missing path: {}", descriptor)
    return default


def normalize_text_advanced(
    text: Any,
    *,
    case_sensitive: bool = False,
    strip: bool = True,
    collapse_whitespace: bool = False,
) -> Optional[str]:
    """Flexible text normalization helper used across metrics."""

    if text is None:
        return None
    s = str(text)
    if strip:
        s = s.strip()
    if collapse_whitespace:
        s = re.sub(r"\s+", " ", s)
    if not case_sensitive:
        s = s.lower()
    return s


def ensure_list_of_strings(value: Any, ignore_none: bool = True) -> list[str]:
    """Coerce input into a list of strings, dropping None if requested."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value] if (value or not ignore_none) else []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if (v is not None or not ignore_none)]
    return [str(value)]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein distance for lightweight text similarity checks."""

    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def strip_thought_tags(text: Any) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""

    raw = "" if text is None else str(text)
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()


def flatten_numeric_list(values: Any) -> list[float]:
    """Flatten nested numeric containers into a flat float list."""

    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        flat: list[float] = []
        for item in values:
            flat.extend(flatten_numeric_list(item))
        return flat
    try:
        return [float(values)]
    except (TypeError, ValueError):
        return []

def get_text_content_of_first_predict_result(sample_dict):
    try:
        ret = sample_dict['predict_result'][0]['message']['content'][0]['text']
        return ret
    except Exception as e:
        logger.warning(f'[warning]{e}')        
        return None

def get_sample_options(sample_dict):
    try:
        sample_dict.get("options")
    except Exception as e:
        logger.warning(f'[warning]{e}')        
        return None

def get_sample_label(sample_dict):
    return sample_dict.get('label')

def get_first_reference(sample_dict):
    try:
        references= sample_dict.get('references')
        if len(references) >= 1:
            return references[0]
    except Exception as e:
        logger.warning(f'[warning]{e}')
        return None

__all__ = [
    "get_text_content_of_first_predict_result",
    "get_sample_label",
    "get_sample_options",
    "get_first_reference",
    "extract_field",
    "walk_path",
    "normalize_text_advanced",
    "ensure_list_of_strings",
    "levenshtein_distance",
    "strip_thought_tags",
    "flatten_numeric_list",
]
