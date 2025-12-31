"""Builtin runtime substitution helpers (Phase 1).

This module focuses on guarding runtime parameter naming, especially when
force-merging multi-task configs. It will be extended in later phases to
support full template compilation.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

PLACEHOLDER_PATTERN = re.compile(r"^\$\{([^}]+)\}$")
INLINE_PATTERN = re.compile(r"\$\{([^}]+)\}")


class RuntimeParameterError(ValueError):
    """Raised when runtime parameter substitution fails."""


def substitute_runtime_placeholders(
    definition: Any,
    runtime_params: Mapping[str, Any],
    *,
    force_merge: bool = False,
    task_ids: Sequence[str] | None = None,
) -> Any:
    """Recursively replace ${runtime.*} placeholders with concrete values.

    When ``force_merge`` is True, any ``runtime.tasks`` reference must include a
    task id, and if ``task_ids`` are provided, the id must be one of them.
    """

    return _substitute(
        deepcopy(definition),
        runtime_params,
        force_merge=force_merge,
        task_ids=task_ids,
        path_stack=["definition"],
    )


def _substitute(
    obj: Any,
    runtime_params: Mapping[str, Any],
    *,
    force_merge: bool,
    task_ids: Sequence[str] | None,
    path_stack: list[str],
) -> Any:
    field_path = _format_path(path_stack)
    if isinstance(obj, str):
        match = PLACEHOLDER_PATTERN.match(obj)
        if match:
            path = match.group(1)
            return _resolve_runtime_path(
                path,
                runtime_params,
                force_merge=force_merge,
                task_ids=task_ids,
                path_stack=path_stack,
            )
        # 部分插值：保持字符串类型，将占位符替换为 str(value)
        def _replace(m: re.Match[str]) -> str:
            placeholder = m.group(1)
            value = _resolve_runtime_path(
                placeholder,
                runtime_params,
                force_merge=force_merge,
                task_ids=task_ids,
                path_stack=path_stack,
            )
            return str(value)

        return INLINE_PATTERN.sub(_replace, obj)

    if isinstance(obj, Mapping):
        replaced = {}
        for key, value in obj.items():
            child_path_stack = path_stack + [str(key)]
            replaced[key] = _substitute(
                value,
                runtime_params,
                force_merge=force_merge,
                task_ids=task_ids,
                path_stack=child_path_stack,
            )
        return replaced

    if isinstance(obj, (list, tuple)):
        return type(obj)(
            _substitute(
                item,
                runtime_params,
                force_merge=force_merge,
                task_ids=task_ids,
                path_stack=path_stack + [f"[{idx}]"],
            )
            for idx, item in enumerate(obj)
        )

    return obj


def _resolve_runtime_path(
    path: str,
    runtime_params: Mapping[str, Any],
    *,
    force_merge: bool,
    task_ids: Sequence[str] | None,
    path_stack: list[str],
) -> Any:
    field_path = _format_path(path_stack)
    if not path.startswith("runtime."):
        raise RuntimeParameterError(
            f"Only runtime.* placeholders are supported (got '{path}') at field '{field_path}'"
        )

    tokens = path.split(".")
    if force_merge and len(tokens) >= 2 and tokens[1] == "tasks":
        if len(tokens) < 3:
            raise RuntimeParameterError(
                f"runtime.tasks path must include an explicit task id when force_merge is enabled "
                f"(field '{field_path}', placeholder '${{{path}}}')"
            )
        task_id = tokens[2]
        if task_ids is not None and task_id not in task_ids:
            known = ", ".join(task_ids)
            raise RuntimeParameterError(
                f"runtime.tasks path references unknown task id '{task_id}' (known tasks: {known}) "
                f"at field '{field_path}', placeholder '${{{path}}}'"
            )

    cursor: Any = runtime_params
    for token in tokens[1:]:
        if not isinstance(cursor, Mapping) or token not in cursor:
            raise RuntimeParameterError(
                f"runtime path '${{{path}}}' missing key '{token}' at field '{field_path}'"
            )
        cursor = cursor[token]
    return deepcopy(cursor)


def _format_path(stack: list[str]) -> str:
    if not stack:
        return "<root>"
    # Merge index tokens "[i]" into previous element when possible for readability.
    parts: list[str] = []
    for token in stack:
        if token.startswith("[") and parts:
            parts[-1] = f"{parts[-1]}{token}"
        else:
            parts.append(token)
    return ".".join(parts)


__all__ = [
    "RuntimeParameterError",
    "substitute_runtime_placeholders",
]
