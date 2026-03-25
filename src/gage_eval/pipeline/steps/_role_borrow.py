"""Compatibility helpers for borrowing runtime roles from different managers."""

from __future__ import annotations

import inspect
from typing import Any


def borrow_role_with_optional_context(
    role_manager: Any,
    adapter_id: str | None,
    *,
    execution_context: Any = None,
):
    """Borrow a role while remaining compatible with legacy test stubs."""

    if execution_context is None:
        return role_manager.borrow_role(adapter_id)
    borrow_role = role_manager.borrow_role
    try:
        signature = inspect.signature(borrow_role)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        if "execution_context" in signature.parameters:
            return borrow_role(adapter_id, execution_context=execution_context)
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return borrow_role(adapter_id, execution_context=execution_context)
    return borrow_role(adapter_id)
