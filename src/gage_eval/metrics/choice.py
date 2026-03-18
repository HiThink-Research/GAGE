"""Choice utilities."""

from __future__ import annotations

import re
from typing import Iterable


DEFAULT_CHOICE_PATTERNS: tuple[str, ...] = (
    r"\\boxed\{\s*([A-Ea-e])\s*\}",
    r"<answer>\s*([A-Ea-e])",
    r"\b([A-Ea-e])\b",
    r"(?i)(?:^|[^a-zA-Z])([A-Z])(?=[^a-zA-Z]|$)",
)
CASE_SENSITIVE_CHOICE_PATTERNS: tuple[str, ...] = (
    r"\\boxed\{\s*([A-E])\s*\}",
    r"<answer>\s*([A-E])",
    r"\b([A-E])\b",
    r"(?:^|[^A-Z])([A-Z])(?=[^A-Z]|$)",
)


def extract_single_choice_letter(
    prediction: str,
    ignore_case: bool = True,
    patterns: Iterable[str] | None = None,
) -> str | None:
    """Extracts the last obvious multiple-choice token from a prediction.

    Args:
        prediction: Model output to inspect.
        ignore_case: Whether default patterns should match lowercase letters.
        patterns: Custom regex patterns to use instead of the built-in defaults.

    Returns:
        The last matched token, or ``None`` when no match is found.
    """

    if patterns is None:
        active_patterns = list(
            DEFAULT_CHOICE_PATTERNS if ignore_case else CASE_SENSITIVE_CHOICE_PATTERNS
        )
    else:
        active_patterns = list(patterns)

    candidates: list[str | tuple[str, ...]] = []
    for pat in active_patterns:
        candidates.extend(re.findall(pat, prediction))

    # Flatten tuple matches returned by regex groups.
    flat: list[str] = []
    for cand in candidates:
        if isinstance(cand, tuple):
            flat.extend([c for c in cand if c])
        elif cand:
            flat.append(cand)

    return flat[-1] if flat else None
