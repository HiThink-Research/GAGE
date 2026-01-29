"""Temporary stable-retro data contract helpers (section 15).

These helpers are scoped to the retro integration only. They are intentionally
not part of the core `ArenaObservation` / `ArenaAction` / `GameResult` API.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, TypedDict


class ObservationView(TypedDict, total=False):
    text: str
    image: Any
    vector: list[Any]


class ObservationContext(TypedDict):
    mode: str
    step: int
    tick: int


class ObservationDict(TypedDict, total=False):
    view: ObservationView
    legal_actions: list[str]
    context: ObservationContext
    active_player: str
    extra: dict[str, Any]


class ActionDict(TypedDict, total=False):
    player: str
    move: str
    raw: str
    hold_ticks: int


class ResultDict(TypedDict, total=False):
    winner: str
    scores: dict[str, float]
    status: str
    reason: str
    replay_path: str
    metrics: dict[str, Any]


def build_observation_dict(
    *,
    view_text: str,
    legal_actions: Sequence[str],
    active_player: str,
    tick: int,
    step: int,
    info: Mapping[str, Any],
    extra: Optional[dict[str, Any]] = None,
) -> ObservationDict:
    """Build a section-15 observation dict.

    Args:
        view_text: Text payload for `view.text`.
        legal_actions: Legal macro actions.
        active_player: Active player id.
        tick: Tick counter.
        step: Decision counter.
        info: Per-tick stable-retro info payload (stored in extra.info).
        extra: Optional extra fields to merge into extra.

    Returns:
        ObservationDict that matches section 15.1.
    """

    merged_extra: dict[str, Any] = {"info": dict(info)}
    if extra:
        merged_extra.update(extra)
    return {
        "view": {"text": str(view_text)},
        "legal_actions": [str(item) for item in legal_actions],
        "context": {"mode": "tick", "step": int(step), "tick": int(tick)},
        "active_player": str(active_player),
        "extra": merged_extra,
    }


def build_action_dict(
    *,
    player: str,
    move: str,
    raw: str,
    hold_ticks: Optional[int],
) -> ActionDict:
    """Build a section-15 action dict.

    Args:
        player: Player id.
        move: Normalized macro move.
        raw: Raw model output string.
        hold_ticks: Optional number of ticks to hold the move.

    Returns:
        ActionDict that matches section 15.2.
    """

    payload: ActionDict = {"player": str(player), "move": str(move), "raw": str(raw)}
    if hold_ticks is not None:
        payload["hold_ticks"] = int(hold_ticks)
    return payload


def build_result_dict(
    *,
    status: str,
    reason: Optional[str],
    winner: Optional[str],
    replay_path: Optional[str],
    scores: Optional[Mapping[str, float]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
) -> ResultDict:
    """Build a section-15 result dict.

    Args:
        status: One of win/loss/draw/terminated.
        reason: Optional reason string.
        winner: Optional winner player id.
        replay_path: Optional replay schema path.
        scores: Optional score mapping.
        metrics: Optional metrics mapping.

    Returns:
        ResultDict that matches section 15.3.
    """

    payload: ResultDict = {"status": str(status), "reason": str(reason or "")}
    if winner:
        payload["winner"] = str(winner)
    if replay_path:
        payload["replay_path"] = str(replay_path)
    if scores:
        payload["scores"] = {str(k): float(v) for k, v in dict(scores).items()}
    if metrics:
        payload["metrics"] = dict(metrics)
    return payload

