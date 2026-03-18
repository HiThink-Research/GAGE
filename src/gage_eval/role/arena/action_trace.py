"""Helpers for semantic action display values in arena traces."""

from __future__ import annotations

from typing import Any, Optional

from gage_eval.role.arena.types import ArenaObservation

TRACE_ACTION_APPLIED_KEY = "trace_action_applied"


def attach_trace_action_applied(
    metadata: Optional[dict[str, Any]],
    *,
    observation: ArenaObservation,
    move: Any,
) -> dict[str, Any]:
    """Attach a semantic action label for trace serialization when available.

    Args:
        metadata: Existing action metadata.
        observation: Observation used to build the action.
        move: Canonical action move that will be executed by the environment.

    Returns:
        A metadata copy that includes a trace-friendly action label when the
        observation exposes one.
    """

    result = dict(metadata or {})
    display_value = resolve_observation_action_label(observation, move)
    if display_value:
        result[TRACE_ACTION_APPLIED_KEY] = display_value
    return result


def resolve_trace_action_applied(action_move: Any, metadata: Optional[dict[str, Any]]) -> Any:
    """Resolve the trace display value for an applied action.

    Args:
        action_move: Canonical move value used for environment execution.
        metadata: Action metadata that may contain a semantic display label.

    Returns:
        The semantic label when present, otherwise the canonical move value.
    """

    if isinstance(metadata, dict):
        display_value = metadata.get(TRACE_ACTION_APPLIED_KEY)
        if display_value not in (None, ""):
            return display_value
    return action_move


def sanitize_trace_action_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Drop internal trace-only keys from serialized action metadata.

    Args:
        metadata: Action metadata to be serialized.

    Returns:
        Metadata without internal display-only keys.
    """

    result = dict(metadata or {})
    result.pop(TRACE_ACTION_APPLIED_KEY, None)
    return result


def resolve_observation_action_label(observation: ArenaObservation, move: Any) -> Optional[str]:
    """Resolve a semantic action label from observation-provided action mapping.

    Args:
        observation: Observation that may carry an action mapping.
        move: Canonical move value.

    Returns:
        The mapped action label when present, otherwise ``None``.
    """

    if move is None:
        return None
    raw_mapping = _extract_action_mapping(observation)
    if not isinstance(raw_mapping, dict):
        return None
    move_key = str(move).strip()
    if not move_key:
        return None
    label = raw_mapping.get(move_key)
    if label is None:
        return None
    label_text = str(label).strip()
    return label_text or None


def _extract_action_mapping(observation: ArenaObservation) -> Optional[dict[str, Any]]:
    metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
    action_mapping = metadata.get("action_mapping")
    if isinstance(action_mapping, dict):
        return action_mapping

    prompt = observation.prompt
    if prompt is None or not isinstance(prompt.payload, dict):
        return None
    prompt_mapping = prompt.payload.get("action_mapping")
    if isinstance(prompt_mapping, dict):
        return prompt_mapping
    vizdoom_payload = prompt.payload.get("vizdoom")
    if isinstance(vizdoom_payload, dict) and isinstance(vizdoom_payload.get("action_mapping"), dict):
        return vizdoom_payload.get("action_mapping")
    return None
