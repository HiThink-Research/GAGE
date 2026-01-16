"""Choice utilities."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional, Callable, Literal, List, NamedTuple


def extract_single_choice_letter(prediction: str) -> Optional[str]:
    """Extract the last obvious multiple-choice letter from a model prediction."""

    patterns = [
        r"\\boxed\{\s*([A-Ea-e])\s*\}",
        r"<answer>\s*([A-Ea-e])",
        r"\b([A-Ea-e])\b",
        r"(?i)(?:^|[^a-zA-Z])([A-Z])(?=[^a-zA-Z]|$)"
    ]
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(re.findall(pat, prediction))
    # Flatten tuple matches returned by regex groups.
    flat: list[str] = []
    for cand in candidates:
        if isinstance(cand, tuple):
            flat.extend([c for c in cand if c])
        elif cand:
            flat.append(cand)
    return flat[-1] if flat else None