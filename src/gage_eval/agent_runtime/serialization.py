from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def to_json_compatible(value: Any) -> Any:
    """Convert runtime values into a JSON-compatible structure.

    The runtime can temporarily carry helper objects such as verifier adapters,
    sandbox providers, or dataclass-backed handles. Before these values are
    persisted into artifacts or propagated through model output, we normalize
    them into plain JSON-friendly containers.

    Args:
        value: Arbitrary runtime value.

    Returns:
        A recursively normalized value that can be serialized with `json.dumps`.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return to_json_compatible(value.value)
    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_compatible(item) for item in value]
    if is_dataclass(value):
        return to_json_compatible(asdict(value))
    if isinstance(value, BaseException):
        return {
            "object_type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "message": str(value),
        }

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_json_compatible(to_dict())
        except TypeError:
            pass

    return {
        "object_type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
    }
