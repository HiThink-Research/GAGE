"""Shared helpers for normalizing role backend error payloads."""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace


def raise_for_backend_error(
    *,
    event_prefix: str,
    step_label: str,
    adapter_id: Optional[str],
    output: Any,
    trace: ObservabilityTrace,
) -> None:
    """Raises a normalized RuntimeError for role backend error payloads.

    Args:
        event_prefix: Trace event prefix such as ``"inference"`` or ``"judge"``.
        step_label: Human-readable step label used in log messages.
        adapter_id: Adapter identifier associated with the role invocation.
        output: Raw role output to inspect.
        trace: Observability trace that receives the normalized error event.

    Raises:
        RuntimeError: If the output contains a truthy ``error`` field.
    """

    if not isinstance(output, dict) or not output.get("error"):
        return

    error_text = str(output.get("error"))
    trace.emit(
        f"{event_prefix}_error",
        payload={
            "adapter_id": adapter_id,
            "error_type": "backend_error",
            "failure_reason": "backend_returned_error",
            "error": error_text,
        },
    )
    logger.error(
        "{} failed adapter_id={} error_type=backend_error error={}",
        step_label,
        adapter_id,
        error_text,
    )
    raise RuntimeError(f"{event_prefix} backend returned error: {error_text}")
