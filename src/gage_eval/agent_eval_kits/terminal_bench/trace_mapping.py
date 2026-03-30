"""Terminal benchmark trace mapping helpers."""

from __future__ import annotations

from typing import Any, Iterable


def _normalize_event(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        payload = dict(event)
    elif hasattr(event, "to_dict"):
        payload = dict(event.to_dict())
    else:
        payload = {
            "name": getattr(event, "name", "trace_event"),
            "payload": dict(getattr(event, "payload", {}) or {}),
            "level": getattr(event, "level", "info"),
        }
    payload.setdefault("name", "trace_event")
    payload.setdefault("payload", {})
    payload.setdefault("level", "info")
    return payload


def normalize_trace_events(trace_source: Any) -> tuple[dict[str, Any], ...]:
    """Normalize a trace source into serializable events."""

    if trace_source is None:
        return ()
    if isinstance(trace_source, dict):
        if isinstance(trace_source.get("events"), list):
            return tuple(_normalize_event(event) for event in trace_source["events"])
        return (_normalize_event(trace_source),)
    if isinstance(trace_source, (list, tuple)):
        return tuple(_normalize_event(event) for event in trace_source)
    events = getattr(trace_source, "events", None)
    if isinstance(events, Iterable):
        return tuple(_normalize_event(event) for event in events)
    return (_normalize_event(trace_source),)
