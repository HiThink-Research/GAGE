"""Default prompt mappings for role adapters."""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_PROMPT_BY_HELPER_MODE: dict[str, str] = {
    "answer_cleaner": "helper/answer_cleaner@v1",
}


def resolve_prompt_id_for_adapter(
    adapter_id: str,
    role_type: str,
    params: dict[str, Any],
) -> Optional[str]:
    """Resolve the default prompt id for a role adapter.

    Args:
        adapter_id: The adapter identifier (unused; kept for tracing/debugging).
        role_type: The adapter role type.
        params: Adapter params, used to detect mode-specific defaults.

    Returns:
        The resolved prompt id, or None when no default applies.
    """

    if role_type != "helper_model":
        return None
    mode = params.get("mode")
    if not isinstance(mode, str):
        return None
    return DEFAULT_PROMPT_BY_HELPER_MODE.get(mode)


__all__ = ["DEFAULT_PROMPT_BY_HELPER_MODE", "resolve_prompt_id_for_adapter"]
