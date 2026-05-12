from __future__ import annotations

from typing import Any

from gage_eval.agent_eval_kits.common import build_noop_trace_mapping

map_trace_payload = build_noop_trace_mapping

_ORDERED_TOOL_EVENTS = (
    "model.request",
    "model.response",
    "tool.call.normalized",
    "tool.result",
    "tool.result.injected",
)


def evaluate_tau2_trace_order(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize Tau2 trace ordering by turn for native evaluation diagnostics."""

    turns: dict[int, list[str]] = {}
    invalid: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        turn_index = payload.get("turn_index")
        if turn_index is None and event.get("event_type") in {"tool.result", "tool.result.injected"}:
            turn_index = _infer_latest_turn(turns)
        if turn_index is None:
            continue
        try:
            turn = int(turn_index)
        except (TypeError, ValueError):
            invalid.append({"reason": "invalid_turn_index", "event": event})
            continue
        turns.setdefault(turn, []).append(str(event.get("event_type") or event.get("event") or ""))

    summaries: list[dict[str, Any]] = []
    for turn, names in sorted(turns.items()):
        expected_positions = {name: index for index, name in enumerate(_ORDERED_TOOL_EVENTS)}
        last = -1
        for name in names:
            if name not in expected_positions:
                continue
            position = expected_positions[name]
            if position < last:
                invalid.append({"reason": "event_order", "turn_index": turn, "events": list(names)})
                break
            last = position
        summaries.append({"turn_index": turn, "events": names})
    return {
        "valid": not invalid,
        "turn_count": len(summaries),
        "turns": summaries,
        "invalid": invalid,
    }


def build_tool_trace_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    summary = evaluate_tau2_trace_order(events)
    return {
        "valid": summary["valid"],
        "turn_count": summary["turn_count"],
        "turns": summary["turns"],
    }


def _infer_latest_turn(turns: dict[int, list[str]]) -> int | None:
    if not turns:
        return None
    return max(turns)
