"""Helpers for forwarding role-produced observability events to traces."""

from __future__ import annotations

from typing import Any

from gage_eval.observability.trace import ObservabilityTrace


def emit_observability_events(trace: ObservabilityTrace, sample: dict, output: Any) -> None:
    """Forward role output observability events to the pipeline trace."""

    if not isinstance(output, dict):
        return
    events = output.get("observability_events")
    if not isinstance(events, list):
        return
    sample_id = sample.get("id") if isinstance(sample, dict) else None
    for item in events:
        if not isinstance(item, dict):
            continue
        name = item.get("event")
        payload = item.get("payload")
        if not name or not isinstance(payload, dict):
            continue
        trace.emit(str(name), payload, sample_id=sample_id)
